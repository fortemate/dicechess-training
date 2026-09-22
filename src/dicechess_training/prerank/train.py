"""Train the pre-ranker, and say whether it beat the ordering the engine already ships.

The protocol is `docs/prerank/protocol-v1.json` and was written before this ran. Two of its rules
shape everything here.

**Only groups larger than the shortlist are measured.** In a group of 20 candidates a shortlist of
48 keeps all of them, so the ranker cannot be wrong and the group would contribute a guaranteed
success. In the first corpus that is 45% of groups — enough to turn a bad result into a good
headline, which is exactly why the protocol fixes the subset before anyone sees a number.

**A run that does not beat `material_diff` is a negative result.** That column is the engine's own
shipped pre-ranker, `ExpectimaxSearch.materialBatch`, and it costs nothing because it is already
one of the nine features. The ablation of #17 produced arms worse than a constant and only an
admissibility floor caught it; this is the same floor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path

import numpy as np
import torch

from dicechess_training.prerank.dataset import (
    Corpus,
    all_gains,
    evaluated,
    load_corpus,
    standardisation,
    trainable,
)
from dicechess_training.prerank.model import PreRankMLP, listwise_loss

PROTOCOL = "playground-prerank-v1"
DEFAULT_K = 48
DEFAULT_SEED = 11

#: Shortlist widths worth reporting. 48 is the champion's production `candidateLimit` and the
#: protocol's primary; 8 and 16 are what the engine's ONNX search uses, and they are where an
#: ordering matters most, because a wide shortlist forgives a bad one.
REPORTED_K = (8, 16, 48)

#: Resamples for the paired interval. A margin over a baseline on a few hundred groups is not a
#: result without one: this programme has already been caught once quoting a plateau that a
#: +/-22pp interval could not have shown.
#:
#: 1000 because the protocol preregistered 1000. It is the kind of number that looks free to
#: raise afterwards, which is exactly why a preregistration names it — a knob turned after seeing
#: a result is not a knob, and an interval is the last place to allow one.
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 13


@dataclass(frozen=True)
class Hyperparameters:
    """Everything the protocol fixes, in one place so a run can record what it ran with."""

    hidden_dims: tuple[int, ...] = (32, 32)
    learning_rate: float = 1e-3
    batch_groups: int = 64
    max_epochs: int = 40
    patience: int = 5
    k: int = DEFAULT_K
    seed: int = DEFAULT_SEED


#: The protocol's settings, as one value rather than a call in a default argument.
DEFAULTS = Hyperparameters()


def ranking_metrics(
    corpus: Corpus, scores: np.ndarray, groups: np.ndarray, k: int
) -> dict[str, float]:
    """How often the shortlist keeps a turn the teacher would have chosen.

    `recall_at_k` is the question the seam actually asks: after the cheap ranker has cut the list
    to `k`, is a best candidate still in it? `rank1` is the harder version — did the ranker put
    one first.

    Ties are settled by target *value*, never by index. A group can hold several candidates the
    teacher scored identically, and any of them is a correct answer; scoring against a single
    chosen index would mark a right answer wrong whenever the teacher was indifferent, which in
    the first corpus is most groups somewhere in their list.
    """
    hits = 0
    firsts = 0
    for group in groups:
        rows = corpus.rows(group)
        targets = corpus.targets[rows]
        best = targets.max()
        order = np.argsort(-scores[rows], kind="stable")
        if (targets[order[:k]] == best).any():
            hits += 1
        if targets[order[0]] == best:
            firsts += 1
    total = max(len(groups), 1)
    return {"recall_at_k": hits / total, "rank1": firsts / total, "groups": float(len(groups))}


def hit_vector(corpus: Corpus, scores: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    """Per group, whether a shortlist of `k` under this ordering keeps a best candidate.

    The vector rather than its mean, because the comparison against a baseline is *paired*: both
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


def discordance(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """The groups the two orderings disagree about, and an exact test on them.

    A paired comparison of two rankers is decided entirely by the groups where one keeps a best
    candidate and the other does not; the groups they both get right or both get wrong carry no
    information about which is better. Counting those directly gives the same comparison as the
    bootstrap without depending on how many resamples someone drew — which matters here, because
    at one of the reported widths the bootstrap's lower bound lands on either side of zero
    depending on that count.

    The p-value is the exact two-sided binomial on the discordant pairs, which is McNemar's test
    without the chi-squared approximation. The protocol's bootstrap stays primary; this is the
    same data with nothing left to a random draw.
    """
    wins = int((candidate & ~reference).sum())
    losses = int((~candidate & reference).sum())
    return {
        "wins": wins,
        "losses": losses,
        "p_value": _two_sided_binomial(wins, wins + losses),
    }


def _two_sided_binomial(successes: int, trials: int) -> float:
    """P(a result at least this extreme) under a fair coin, summed over both tails."""
    if trials == 0:
        return 1.0
    weights = [comb(trials, k) * 0.5**trials for k in range(trials + 1)]
    observed = weights[successes]
    return float(min(1.0, sum(w for w in weights if w <= observed * (1 + 1e-9))))


def paired_interval(
    candidate: np.ndarray,
    reference: np.ndarray,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """A percentile interval on the mean paired difference, resampling whole groups."""
    difference = candidate.astype(float) - reference.astype(float)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            difference[rng.integers(0, len(difference), len(difference))].mean()
            for _ in range(repeats)
        ]
    )
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"delta": float(difference.mean()), "ci_low": float(low), "ci_high": float(high)}


def baseline_scores(corpus: Corpus, name: str, seed: int = DEFAULT_SEED) -> np.ndarray:
    """An ordering to measure the model against.

    `material_diff` is not an arbitrary choice of column: ordering by it *is* the pre-ranker the
    engine ships in `ExpectimaxSearch.materialBatch`, so beating it is the least a learned ranker
    has to do to be worth serving. `random` is the lower bound, and says how much of any figure is
    the shortlist being generous rather than the ranker being right.
    """
    if name == "material_diff":
        return corpus.features[:, corpus.columns.index("material_diff")].astype(np.float64)
    if name == "random":
        return np.random.default_rng(seed).random(len(corpus.targets))
    raise ValueError(f"unknown baseline {name!r}")


def _batch(corpus: Corpus, gains: np.ndarray, groups: np.ndarray):
    """One batch: the candidates of several groups, concatenated, with their group index."""
    rows = np.concatenate([np.arange(corpus.offsets[g], corpus.offsets[g + 1]) for g in groups])
    index = np.repeat(np.arange(len(groups)), corpus.sizes[groups])
    return (
        torch.from_numpy(corpus.features[rows]),
        torch.from_numpy(gains[rows]),
        torch.from_numpy(index),
    )


def score_all(model: PreRankMLP, corpus: Corpus, batch: int = 65536) -> np.ndarray:
    """Every candidate's score, in corpus order."""
    model.eval()
    out = np.empty(len(corpus.targets), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(out), batch):
            stop = min(start + batch, len(out))
            block = torch.from_numpy(corpus.features[start:stop])
            out[start:stop] = model(block).reshape(-1).numpy()
    return out


def train(corpus: Corpus, hyper: Hyperparameters = DEFAULTS) -> tuple[dict, PreRankMLP, np.ndarray]:
    """One training run: the report, the fitted model, and its score for every candidate."""
    torch.manual_seed(hyper.seed)
    rng = np.random.default_rng(hyper.seed)

    fitting = trainable(corpus, "train")
    if len(fitting) == 0:
        raise ValueError("the training split has no group with an ordering to learn")
    measuring = evaluated(corpus, "validation", hyper.k)
    if len(measuring) == 0:
        # Without this, `ranking_metrics` returns a recall of zero over no groups, the first
        # epoch becomes the best epoch because zero beats the initial sentinel, and the run
        # early-stops and reports a trained model whose validation evidence does not exist.
        raise ValueError(
            f"the validation split has no group larger than the shortlist of {hyper.k}, "
            "so there is nothing the ranker could be measured on"
        )

    gains = all_gains(corpus)
    mean, scale = standardisation(corpus, fitting)
    model = PreRankMLP(
        input_dim=len(corpus.columns),
        hidden_dims=list(hyper.hidden_dims),
        feature_mean=mean,
        feature_scale=scale,
    ).double()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyper.learning_rate)

    history: list[dict] = []
    best = {"recall_at_k": -1.0}
    best_state: dict | None = None
    best_epoch = -1

    for epoch in range(1, hyper.max_epochs + 1):
        model.train()
        shuffled = rng.permutation(fitting)
        total = 0.0
        batches = 0
        for start in range(0, len(shuffled), hyper.batch_groups):
            chunk = shuffled[start : start + hyper.batch_groups]
            features, batch_gains, index = _batch(corpus, gains, chunk)
            optimiser.zero_grad()
            loss = listwise_loss(model(features), batch_gains, index, len(chunk))
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1

        validation = ranking_metrics(corpus, score_all(model, corpus), measuring, hyper.k)
        history.append({"epoch": epoch, "train_loss": total / max(batches, 1), **validation})
        if validation["recall_at_k"] > best["recall_at_k"]:
            best = validation
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= hyper.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    scores = score_all(model, corpus)
    report = {
        "protocol": PROTOCOL,
        "seed": hyper.seed,
        "k": hyper.k,
        "hyperparameters": asdict(hyper),
        "corpus": {
            "groups": corpus.groups,
            "candidates": int(len(corpus.targets)),
            "trained_on": int(len(fitting)),
            "engine_version": corpus.manifest["engine_version"],
            "teacher": corpus.manifest["teacher"]["id"],
            "source_sha256": corpus.manifest["source_sha256"],
            "groups_sha256": corpus.manifest["groups_sha256"],
        },
        "best_epoch": best_epoch,
        "history": history,
        "validation": best,
        "baselines": {
            name: ranking_metrics(
                corpus, baseline_scores(corpus, name, hyper.seed), measuring, hyper.k
            )
            for name in ("material_diff", "random")
        },
    }
    report["admissible"] = bool(
        best["recall_at_k"] > report["baselines"]["material_diff"]["recall_at_k"]
    )
    report["by_shortlist"] = by_shortlist(corpus, scores, hyper.seed)
    return report, model, scores


def by_shortlist(corpus: Corpus, scores: np.ndarray, seed: int, split: str = "validation") -> dict:
    """Recall at each reported width, with a paired interval against the shipped ordering.

    One width is a headline; three are a shape. A wide shortlist forgives a bad ordering — a
    random order already keeps the best candidate half the time at 48 — so a figure quoted at one
    width says as much about the width as about the ranker.
    """
    material = baseline_scores(corpus, "material_diff")
    random_order = baseline_scores(corpus, "random", seed)
    out = {}
    for k in REPORTED_K:
        groups = evaluated(corpus, split, k)
        if len(groups) == 0:
            # Said rather than averaged. The mean of no groups is a NaN, and a NaN in a results
            # table is a number that someone will eventually read as a measurement.
            out[str(k)] = {"groups": 0, "measured": False}
            continue
        learned = hit_vector(corpus, scores, groups, k)
        shipped = hit_vector(corpus, material, groups, k)
        out[str(k)] = {
            "groups": int(len(groups)),
            "measured": True,
            "learned": float(learned.mean()),
            "material_diff": float(shipped.mean()),
            "random": float(hit_vector(corpus, random_order, groups, k).mean()),
            "paired_vs_material": paired_interval(learned, shipped),
            "discordance_vs_material": discordance(learned, shipped),
        }
    return out


def run(directory, destination=None, hyper: Hyperparameters = DEFAULTS) -> dict:
    """Load, train, and write the report and the weights beside each other."""
    corpus = load_corpus(directory)
    report, model, _ = train(corpus, hyper)
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"report-seed-{hyper.seed}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        torch.save(model.state_dict(), destination / f"weights-seed-{hyper.seed}.pt")
    return report
