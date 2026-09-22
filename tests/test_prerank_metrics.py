"""The list metrics, pinned on lists whose right answer can be read off by eye.

NDCG gets the most attention here because it is the one metric that can be wrong quietly: recall
is a yes or no that a reader can check by hand, while a discounted sum divided by its own ideal
will happily return a plausible number from a broken formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from dicechess_training.prerank.dataset import Corpus, target_gains
from dicechess_training.prerank.metrics import (
    REPORTED_WIDTHS,
    discordance,
    hit_vector,
    ndcg_vector,
    paired_interval,
    report,
)

COLUMNS = (
    "p_diff",
    "n_diff",
    "b_diff",
    "r_diff",
    "q_diff",
    "material_diff",
    "total_material",
    "mobility_diff",
    "king_safety_diff",
)


def _corpus(sizes: list[int], splits: list[str], seed: int = 0) -> Corpus:
    rng = np.random.default_rng(seed)
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    total = int(offsets[-1])
    return Corpus(
        features=rng.normal(size=(total, len(COLUMNS))).astype(np.float32),
        targets=rng.normal(size=total),
        offsets=offsets,
        splits=np.asarray(splits),
        columns=COLUMNS,
        manifest={"columns": list(COLUMNS)},
    )


def test_a_perfect_ordering_scores_one() -> None:
    corpus = _corpus([30], ["validation"])
    perfect = corpus.targets.copy()
    assert ndcg_vector(corpus, perfect, np.array([0]), 10) == pytest.approx(1.0)


def test_reversing_the_ordering_scores_worst() -> None:
    corpus = _corpus([30], ["validation"])
    perfect = ndcg_vector(corpus, corpus.targets, np.array([0]), 10)[0]
    reversed_ = ndcg_vector(corpus, -corpus.targets, np.array([0]), 10)[0]
    assert reversed_ < perfect
    assert 0.0 <= reversed_ < 1.0


def test_ndcg_is_between_zero_and_one_whatever_the_ordering() -> None:
    corpus = _corpus([50, 7, 200], ["validation"] * 3, seed=2)
    groups = np.array([0, 1, 2])
    for scores in (
        np.random.default_rng(1).normal(size=len(corpus.targets)),
        np.zeros(len(corpus.targets)),
        corpus.targets,
    ):
        values = ndcg_vector(corpus, scores, groups, 10)
        assert np.all(values >= 0.0) and np.all(values <= 1.0 + 1e-12)


def test_a_group_smaller_than_the_width_is_judged_on_what_it_has() -> None:
    """A list of three cannot fill a shortlist of ten, and must not be penalised for it."""
    corpus = _corpus([3], ["validation"])
    assert ndcg_vector(corpus, corpus.targets, np.array([0]), 10) == pytest.approx(1.0)


def test_a_group_the_teacher_could_not_separate_scores_one_for_any_ordering() -> None:
    """Uniform gains mean every ordering achieves the ideal — which is right: there was nothing
    to get wrong."""
    corpus = _corpus([8], ["validation"])
    corpus.targets[:] = 4.0
    rng = np.random.default_rng(0)
    assert ndcg_vector(corpus, rng.normal(size=8), np.array([0]), 4) == pytest.approx(1.0)


def test_ndcg_matches_the_definition_computed_by_hand() -> None:
    """Four candidates, the model puts the teacher's best second. Worked through explicitly, so a
    plausible-looking wrong formula cannot hide behind a plausible-looking number."""
    corpus = _corpus([4], ["validation"])
    corpus.targets[:] = [10.0, 5.0, 2.0, 1.0]
    # Descending by score the order is 1 (1.0), 2 (0.5), 0 (0.0), 3 (-1.0) — worth writing out,
    # because the first version of this test sorted it wrong in my head and the assertion caught
    # the arithmetic rather than the code.
    scores = np.array([0.0, 1.0, 0.5, -1.0])

    gains = target_gains(corpus.targets)
    discount = 1.0 / np.log2(np.arange(2, 5))
    achieved = gains[1] * discount[0] + gains[2] * discount[1] + gains[0] * discount[2]
    ideal = (np.sort(gains)[::-1][:3] * discount).sum()

    assert ndcg_vector(corpus, scores, np.array([0]), 3) == pytest.approx(achieved / ideal)


def test_ndcg_notices_a_near_miss_that_recall_cannot() -> None:
    """The point of reporting it beside recall: both orderings keep the best candidate inside the
    shortlist, so recall says they are equal, and one of them still put it first.

    The first version of this test pushed the best candidate *out* of the shortlist, which made
    recall differ — so it asserted the opposite of its own name while passing.
    """
    corpus = _corpus([20], ["validation"])
    corpus.targets[:] = np.arange(20)[::-1]
    groups = np.array([0])
    first = np.arange(20)[::-1].astype(float)
    tenth = first.copy()
    tenth[0] = 9.5  # between the 9th and 10th scores: last place inside the kept ten, not outside

    assert hit_vector(corpus, first, groups, 10)[0]
    assert hit_vector(corpus, tenth, groups, 10)[0], "both orderings must keep the best candidate"
    assert ndcg_vector(corpus, first, groups, 10)[0] > ndcg_vector(corpus, tenth, groups, 10)[0]


def test_the_exact_test_survives_a_corpus_large_enough_to_overflow_a_float() -> None:
    """`comb(trials, k) * 0.5 ** trials` raises OverflowError above about 1,030 trials, and the
    trials here are the groups two orderings disagree about — which a larger corpus produces."""
    many = np.zeros(1400, dtype=bool)
    many[:900] = True
    result = discordance(many, ~many)
    assert result["wins"] == 900
    assert result["losses"] == 500
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["p_value"] < 1e-20


# Each width can only be wrong in a group larger than it, so they do not share a denominator.
# Reporting them against one would make rank-1 look harder than it is and recall-at-48 easier.
def test_every_width_is_measured_on_its_own_groups() -> None:
    corpus = _corpus([2, 3, 12, 30, 100], ["validation"] * 5, seed=3)
    out = report(corpus, np.random.default_rng(0).random(len(corpus.targets)))
    assert out["1"]["groups"] == 5
    assert out["2"]["groups"] == 4
    assert out["8"]["groups"] == 3
    assert out["24"]["groups"] == 2
    assert out["48"]["groups"] == 1


def test_rank_one_and_rank_two_are_the_narrowest_widths() -> None:
    """The definition of done asks for rank-1 and rank-2 hit rates; they are recall at those
    widths, on their own groups, and a wider shortlist can never do worse."""
    corpus = _corpus([40] * 12, ["validation"] * 12, seed=6)
    scores = corpus.targets + np.random.default_rng(4).normal(scale=0.5, size=len(corpus.targets))
    out = report(corpus, scores)
    recalls = [out[str(k)]["recall_at_k"] for k in REPORTED_WIDTHS if out[str(k)].get("measured")]
    assert recalls == sorted(recalls), recalls
    assert out["1"]["recall_at_k"] <= out["2"]["recall_at_k"]


def test_a_reference_ordering_brings_paired_evidence() -> None:
    corpus = _corpus([60] * 20, ["validation"] * 20, seed=7)
    scores = corpus.targets.copy()
    reference = np.random.default_rng(5).random(len(corpus.targets))
    out = report(corpus, scores, reference=reference)
    width = out["8"]
    assert width["recall_at_k"] > width["reference_recall_at_k"]
    assert width["ndcg_at_k"] > width["reference_ndcg_at_k"]
    assert width["recall_vs_reference"]["ci_low"] > 0
    assert width["recall_discordance"]["wins"] > width["recall_discordance"]["losses"]
    assert width["ndcg_vs_reference"]["delta"] > 0


def test_a_width_with_no_groups_is_reported_as_unmeasured() -> None:
    corpus = _corpus([3, 4], ["validation", "validation"])
    out = report(corpus, np.zeros(len(corpus.targets)))
    assert out["48"] == {"groups": 0, "measured": False}
    assert out["1"]["measured"] is True


def test_an_empty_comparison_has_no_interval_rather_than_a_nan() -> None:
    assert paired_interval(np.array([], dtype=bool), np.array([], dtype=bool)) == {
        "delta": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
    }
