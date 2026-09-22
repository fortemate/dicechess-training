"""How a candidate ordering is judged, and on which groups.

Split out of `train` because the measurement outlives any one training run: the same numbers are
wanted for an ordering that is already deployed, for a baseline that is a single feature column,
and for an artifact somebody exported last month.

## Every width has its own denominator

A ranker can only be wrong in a group with more candidates than the shortlist keeps. At a
shortlist of 48 that is 264 of the first corpus's 489 validation groups; at a shortlist of 1 it is
473. Measuring them all against one denominator would make a rank-1 figure look harder than it is
and a recall-at-48 figure easier, so each width is measured on `size > k` and the group count is
reported beside every number.

## Ties are settled by value, never by index

A group can hold several candidates the teacher scored identically, and any of them is a right
answer. Scoring against one chosen index would mark a right answer wrong whenever the teacher was
indifferent — which in the first corpus is most groups, somewhere down their list.
"""

from __future__ import annotations

from math import comb

import numpy as np

from dicechess_training.prerank.dataset import Corpus, evaluated, target_gains

#: Widths worth reporting. 1 and 2 are the rank-1 and rank-2 hit rates; 8 and 24 are what the
#: deployed ONNX bots run; 48 is the champion's `candidateLimit` and the protocol's primary; 16 is
#: where the engine's own scaladoc measured its width step.
REPORTED_WIDTHS = (1, 2, 8, 16, 24, 48)

BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 13


def hit_vector(corpus: Corpus, scores: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    """Per group, whether a shortlist of `k` under this ordering keeps a best candidate.

    The vector rather than its mean, because a comparison against a baseline is *paired*: both
    orderings are measured on the same groups, and the interval on their difference is much
    tighter than two independent intervals would suggest.
    """
    out = np.zeros(len(groups), dtype=bool)
    for position, group in enumerate(groups):
        rows = corpus.rows(group)
        targets = corpus.targets[rows]
        order = np.argsort(-scores[rows], kind="stable")
        out[position] = bool((targets[order[:k]] == targets.max()).any())
    return out


def ndcg_vector(corpus: Corpus, scores: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    """Per group, normalised discounted cumulative gain at `k`.

    The relevance of a candidate is the same gain the loss is trained against — `1 / rank**2` on
    the teacher's ordering, normalised — so this metric and the objective agree about what a good
    list looks like instead of quietly optimising one thing and reporting another.

    Where recall answers "did a best candidate survive the cut", this answers "how good is the cut
    as a whole": it moves when the second and third best move, and it distinguishes a ranker that
    put the best candidate first from one that scraped it in at position 48. That is the list
    metric the definition of done asks for, and it is why it is reported beside recall rather than
    instead of it.
    """
    out = np.zeros(len(groups), dtype=float)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    for position, group in enumerate(groups):
        rows = corpus.rows(group)
        gains = target_gains(corpus.targets[rows])
        order = np.argsort(-scores[rows], kind="stable")
        width = min(k, len(gains))
        achieved = float((gains[order[:width]] * discount[:width]).sum())
        ideal = float((np.sort(gains)[::-1][:width] * discount[:width]).sum())
        out[position] = achieved / ideal if ideal > 0 else 1.0
    return out


def discordance(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """The groups the two orderings disagree about, and an exact test on them.

    A paired comparison of two rankers is decided entirely by the groups where one keeps a best
    candidate and the other does not; groups they both get right, or both get wrong, carry no
    information about which is better. Counting those gives the same comparison as the bootstrap
    without depending on how many resamples someone drew.

    The p-value is the exact two-sided binomial on the discordant pairs, which is McNemar's test
    without the chi-squared approximation.
    """
    wins = int((candidate & ~reference).sum())
    losses = int((~candidate & reference).sum())
    return {"wins": wins, "losses": losses, "p_value": _two_sided_binomial(wins, wins + losses)}


def _two_sided_binomial(successes: int, trials: int) -> float:
    """P(a result at least this extreme) under a fair coin, summed over both tails.

    The weights stay exact integers until the very end. `comb(trials, k)` is a Python int of
    arbitrary size, and multiplying it by `0.5 ** trials` converts it to a float first — which
    raises `OverflowError` above about 1,030 trials. That is not hypothetical at a larger corpus:
    the trials here are the groups two orderings disagree about, and this page's own conclusion is
    that a larger corpus is what the measurement needs. Summing integers and dividing once at the
    end has no such ceiling, because Python divides two large integers by scaling rather than by
    converting them.
    """
    if trials == 0:
        return 1.0
    weights = [comb(trials, k) for k in range(trials + 1)]
    observed = weights[successes]
    return float(min(1.0, sum(w for w in weights if w <= observed) / 2**trials))


def paired_interval(
    candidate: np.ndarray,
    reference: np.ndarray,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """A percentile interval on the mean paired difference, resampling whole groups.

    Works for the boolean recall vectors and for the continuous NDCG ones alike, which is the
    reason it takes arrays rather than a metric name.
    """
    difference = candidate.astype(float) - reference.astype(float)
    if len(difference) == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            difference[rng.integers(0, len(difference), len(difference))].mean()
            for _ in range(repeats)
        ]
    )
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"delta": float(difference.mean()), "ci_low": float(low), "ci_high": float(high)}


def report(
    corpus: Corpus,
    scores: np.ndarray,
    split: str = "validation",
    widths: tuple[int, ...] = REPORTED_WIDTHS,
    reference: np.ndarray | None = None,
) -> dict:
    """Every width's recall and NDCG on its own groups, optionally against a reference ordering.

    `reference` is another ordering measured on exactly the same groups — a deployed model, or a
    single feature column. When it is given, every width also carries the paired interval and the
    discordant counts, because a difference between two orderings on a few hundred groups is not a
    result without them.
    """
    out: dict[str, dict] = {}
    for k in widths:
        groups = evaluated(corpus, split, k)
        if len(groups) == 0:
            out[str(k)] = {"groups": 0, "measured": False}
            continue
        hits = hit_vector(corpus, scores, groups, k)
        ndcg = ndcg_vector(corpus, scores, groups, k)
        width: dict = {
            "groups": int(len(groups)),
            "measured": True,
            "recall_at_k": float(hits.mean()),
            "ndcg_at_k": float(ndcg.mean()),
        }
        if reference is not None:
            reference_hits = hit_vector(corpus, reference, groups, k)
            width["reference_recall_at_k"] = float(reference_hits.mean())
            width["reference_ndcg_at_k"] = float(ndcg_vector(corpus, reference, groups, k).mean())
            width["recall_vs_reference"] = paired_interval(hits, reference_hits)
            width["recall_discordance"] = discordance(hits, reference_hits)
            width["ndcg_vs_reference"] = paired_interval(
                ndcg, ndcg_vector(corpus, reference, groups, k)
            )
        out[str(k)] = width
    return out


#: Why no calibration figure appears anywhere above.
#:
#: The definition of done asks for calibration *where meaningful*, and for a ranker it is not.
#: An ordering is invariant to every monotone transform of the scores, so there is no quantity a
#: calibration curve could be drawn against: the model is never asked how good a candidate is, only
#: which of two is better. `contracts.prerank` makes the same point structurally by refusing a
#: `calibration` block on this role — a model that needed one would be a value model wearing the
#: wrong name.
#:
#: What people usually want from calibration here — "how good is the whole list, not just the top
#: pick" — is what `ndcg_at_k` answers, which is why it is reported at every width.
CALIBRATION = "not meaningful for an ordering; see the note in this module"
