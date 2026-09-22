"""What a listwise objective has to be true of, before it is pointed at a real corpus.

Three kinds of test. The target distribution has properties that were *chosen* — top-heavy,
scale-free, tie-honest — and each is asserted rather than described. The loss is a segment
softmax written by hand, so it is checked against the same quantity computed group by group with
`torch.log_softmax`. And the metrics decide whether the whole programme is worth continuing, so
they are pinned on lists whose right answer can be read off by eye.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dicechess_training.prerank.dataset import (
    TERMINAL_TARGET,
    Corpus,
    all_gains,
    evaluated,
    standardisation,
    target_gains,
    trainable,
)
from dicechess_training.prerank.model import PreRankMLP, listwise_loss
from dicechess_training.prerank.train import (
    BOOTSTRAP_REPEATS,
    Hyperparameters,
    _two_sided_binomial,
    baseline_scores,
    by_shortlist,
    discordance,
    hit_vector,
    paired_interval,
    ranking_metrics,
    train,
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
    """A corpus whose targets are a known function of the features, so a ranker can find it."""
    rng = np.random.default_rng(seed)
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    total = int(offsets[-1])
    features = rng.normal(size=(total, len(COLUMNS))).astype(np.float32)
    # The signal leans on mobility and king safety more than on material, so ordering by the
    # `material_diff` column alone — the baseline — is a weak proxy and the ranker has something
    # to find. With material dominant the baseline scores a perfect recall and the test would be
    # asking the model to beat a ceiling.
    targets = (300.0 * features[:, 5] + 900.0 * features[:, 7] - 700.0 * features[:, 8]).astype(
        np.float64
    )
    return Corpus(
        features=features,
        targets=targets,
        offsets=offsets,
        splits=np.asarray(splits),
        columns=COLUMNS,
        manifest={
            "engine_version": "0.12.0",
            "teacher": {"id": "hunter-baseline-v1"},
            "source_sha256": "a" * 64,
            "groups_sha256": "b" * 64,
            "columns": list(COLUMNS),
        },
    )


def test_gains_sum_to_one_and_favour_the_best() -> None:
    gains = target_gains(np.array([10.0, 5.0, 1.0]))
    assert gains.sum() == pytest.approx(1.0)
    assert gains[0] > gains[1] > gains[2]


def test_gains_ignore_the_teachers_units() -> None:
    """Targets are in the teacher's private scale and its spread varies between groups by orders
    of magnitude; the target distribution must not."""
    small = target_gains(np.array([3.0, 2.0, 1.0]))
    large = target_gains(np.array([30000.0, 20000.0, 10000.0]))
    shifted = target_gains(np.array([-3.0, -4.0, -5.0]))
    assert small == pytest.approx(large)
    assert small == pytest.approx(shifted)


def test_the_king_capture_sentinel_needs_no_special_case() -> None:
    """`Int.MaxValue` is four orders of magnitude past any evaluation. Under rank it is first."""
    with_sentinel = target_gains(np.array([TERMINAL_TARGET, 900.0, -3.0]))
    ordinary = target_gains(np.array([3.0, 2.0, 1.0]))
    assert with_sentinel == pytest.approx(ordinary)


def test_a_group_the_teacher_could_not_separate_is_a_uniform_target() -> None:
    """4.2% of the first corpus. The model should be asked to be flat there, not to invent an
    order that the teacher did not have."""
    assert target_gains(np.array([7.0, 7.0, 7.0])) == pytest.approx(np.full(3, 1 / 3))


def test_tied_candidates_share_their_mass() -> None:
    gains = target_gains(np.array([9.0, 9.0, 1.0]))
    assert gains[0] == pytest.approx(gains[1])
    assert gains[0] > gains[2]


def test_a_tie_is_not_broken_by_array_order() -> None:
    forward = target_gains(np.array([5.0, 5.0, 5.0, 1.0]))
    backward = target_gains(np.array([1.0, 5.0, 5.0, 5.0]))
    assert sorted(forward) == pytest.approx(sorted(backward))


# The shortlist keeps k candidates and the search then rescores all of them, so the only thing
# that has to survive the cut is the maximum. The target mass therefore belongs inside the cut,
# whatever the list's length — which is why the NDCG discount was measured out of the protocol
# before the model was trained once.
@pytest.mark.parametrize("size", [61, 200, 643, 2420])
def test_almost_all_of_the_target_mass_is_inside_the_shortlist(size: int) -> None:
    gains = target_gains(np.arange(size, dtype=np.float64))
    assert np.sort(gains)[::-1][:48].sum() > 0.98


def test_the_best_candidates_share_does_not_move_with_the_lists_length() -> None:
    """A long list must teach the same lesson about its top few as a short one; `1 / rank` would
    halve the best candidate's weight between these two."""
    short = target_gains(np.arange(61, dtype=np.float64)).max()
    long = target_gains(np.arange(2420, dtype=np.float64)).max()
    assert short == pytest.approx(long, abs=0.01)


def test_the_segment_loss_equals_a_per_group_cross_entropy() -> None:
    """The loss softmaxes many groups of different lengths in one flat tensor. That is worth
    checking against the obvious, slow version rather than trusting the index arithmetic."""
    rng = np.random.default_rng(3)
    sizes = [5, 1, 40, 2]
    scores = torch.tensor(rng.normal(size=sum(sizes)), dtype=torch.float64, requires_grad=True)
    gains = torch.tensor(
        np.concatenate([target_gains(rng.normal(size=size)) for size in sizes]),
        dtype=torch.float64,
    )
    index = torch.tensor(np.repeat(np.arange(len(sizes)), sizes))

    fused = listwise_loss(scores, gains, index, len(sizes))

    slow = []
    at = 0
    for size in sizes:
        chunk = slice(at, at + size)
        slow.append(-(gains[chunk] * torch.log_softmax(scores[chunk], dim=0)).sum())
        at += size
    assert float(fused.detach()) == pytest.approx(float(torch.stack(slow).mean().detach()))


def test_the_loss_is_smallest_when_the_model_agrees_with_the_teacher() -> None:
    gains = torch.tensor(target_gains(np.array([9.0, 5.0, 1.0])))
    index = torch.zeros(3, dtype=torch.int64)
    agreeing = listwise_loss(torch.tensor([3.0, 1.0, -2.0], dtype=torch.float64), gains, index, 1)
    inverted = listwise_loss(torch.tensor([-2.0, 1.0, 3.0], dtype=torch.float64), gains, index, 1)
    flat = listwise_loss(torch.zeros(3, dtype=torch.float64), gains, index, 1)
    assert agreeing < flat < inverted


def test_the_loss_survives_scores_far_from_zero() -> None:
    """Scores are unbounded by design, so the per-group maximum has to be subtracted."""
    gains = torch.tensor(target_gains(np.array([2.0, 1.0])))
    index = torch.zeros(2, dtype=torch.int64)
    loss = listwise_loss(torch.tensor([1e4, -1e4], dtype=torch.float64), gains, index, 1)
    assert torch.isfinite(loss)


def test_a_long_group_counts_as_one_decision() -> None:
    """The mean is over groups, not candidates: a root with 2,420 turns is one choice."""
    gains = torch.tensor(np.concatenate([target_gains(np.arange(2.0)) for _ in range(2)]))
    index = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    one = listwise_loss(scores[:2], gains[:2], torch.zeros(2, dtype=torch.int64), 1)
    two = listwise_loss(scores, gains, index, 2)
    assert float(two) == pytest.approx(float(one))


def test_recall_counts_a_tie_for_best_as_a_hit() -> None:
    """A group can hold several candidates the teacher scored identically, and any of them is a
    right answer. Scoring against one chosen index would mark a right answer wrong."""
    corpus = _corpus([4], ["validation"])
    corpus.targets[:] = [5.0, 5.0, 1.0, 0.0]
    metrics = ranking_metrics(corpus, np.array([0.0, 9.0, 1.0, 2.0]), np.array([0]), k=1)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["rank1"] == 1.0


def test_recall_is_a_miss_when_the_best_falls_outside_the_shortlist() -> None:
    corpus = _corpus([4], ["validation"])
    corpus.targets[:] = [9.0, 1.0, 1.0, 1.0]
    metrics = ranking_metrics(corpus, np.array([-1.0, 5.0, 4.0, 3.0]), np.array([0]), k=2)
    assert metrics["recall_at_k"] == 0.0
    assert metrics["rank1"] == 0.0


def test_only_groups_bigger_than_the_shortlist_are_evaluated() -> None:
    corpus = _corpus([10, 60, 200], ["validation"] * 3)
    assert list(evaluated(corpus, "validation", k=48)) == [1, 2]
    assert list(evaluated(corpus, "validation", k=1000)) == []


def test_a_single_candidate_group_is_not_trained_on() -> None:
    corpus = _corpus([1, 5], ["train", "train"])
    assert list(trainable(corpus, "train")) == [1]


def test_standardisation_uses_the_given_groups_only() -> None:
    """Fitted on the training partition and nowhere else; statistics over everything would leak
    the validation partition into the model."""
    corpus = _corpus([50, 50], ["train", "validation"])
    mean, scale = standardisation(corpus, corpus.of("train"))
    rows = corpus.features[: corpus.offsets[1]]
    assert mean == pytest.approx(rows.mean(axis=0), abs=1e-6)
    assert scale == pytest.approx(rows.std(axis=0), abs=1e-6)


def test_a_constant_feature_does_not_divide_by_zero() -> None:
    corpus = _corpus([30], ["train"])
    corpus.features[:, 6] = 4.0
    _, scale = standardisation(corpus, corpus.of("train"))
    assert scale[6] == 1.0


def test_the_model_takes_raw_features_and_carries_its_scaler() -> None:
    """The serving contract sends the engine's features exactly as the engine made them."""
    mean = np.full(9, 2.0, dtype=np.float32)
    scale = np.full(9, 4.0, dtype=np.float32)
    model = PreRankMLP(9, [8], feature_mean=mean, feature_scale=scale).double()
    assert "feature_mean" in dict(model.named_buffers())
    raw = torch.full((1, 9), 6.0, dtype=torch.float64)
    bare = PreRankMLP(9, [8]).double()
    bare.net.load_state_dict(model.net.state_dict())
    assert float(model(raw)) == pytest.approx(float(bare(torch.ones(1, 9, dtype=torch.float64))))


def test_the_model_does_not_squash_its_output() -> None:
    """An ordering is invariant to monotone transforms, so there is nothing to calibrate — and a
    sigmoid would collapse the distinctions the ranker exists to make."""
    model = PreRankMLP(9, [8]).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(50.0)
        scores = model(torch.randn(64, 9, dtype=torch.float64))
    assert float(scores.max()) > 1.0 or float(scores.min()) < 0.0


def test_statistics_must_be_given_together() -> None:
    with pytest.raises(ValueError, match="together"):
        PreRankMLP(9, [8], feature_mean=np.zeros(9, dtype=np.float32))


def test_the_baseline_is_the_column_the_engine_already_ranks_by() -> None:
    corpus = _corpus([5], ["train"])
    assert baseline_scores(corpus, "material_diff") == pytest.approx(
        corpus.features[:, COLUMNS.index("material_diff")]
    )
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_scores(corpus, "clairvoyance")


def test_training_learns_an_order_the_baseline_does_not_have() -> None:
    """The corpus's targets depend on material *and* two other columns, so a ranker that reads
    all nine has something to find that ordering by material alone cannot.

    This is the mechanism under test, not a claim about the real corpus: it says the loss, the
    batching and the metric are wired up such that learning is possible and visible.
    """
    sizes = [80] * 120
    splits = ["train"] * 100 + ["validation"] * 20
    corpus = _corpus(sizes, splits, seed=5)
    report, _, _ = train(corpus, Hyperparameters(max_epochs=12, patience=12, k=10, seed=11))

    assert report["validation"]["recall_at_k"] > report["baselines"]["material_diff"]["recall_at_k"]
    assert (
        report["baselines"]["material_diff"]["recall_at_k"]
        > report["baselines"]["random"]["recall_at_k"]
    )
    assert report["admissible"] is True
    assert report["history"][-1]["train_loss"] < report["history"][0]["train_loss"]


def test_a_report_records_what_it_was_trained_on() -> None:
    corpus = _corpus([60] * 30, ["train"] * 25 + ["validation"] * 5, seed=2)
    report, _, _ = train(corpus, Hyperparameters(max_epochs=2, k=10, seed=23))
    assert report["protocol"] == "playground-prerank-v1"
    assert report["seed"] == 23
    assert report["corpus"]["groups_sha256"] == "b" * 64
    assert report["corpus"]["teacher"] == "hunter-baseline-v1"
    assert set(report["baselines"]) == {"material_diff", "random"}


def test_all_gains_are_computed_group_by_group() -> None:
    corpus = _corpus([3, 4], ["train", "train"])
    gains = all_gains(corpus)
    assert gains[:3].sum() == pytest.approx(1.0)
    assert gains[3:].sum() == pytest.approx(1.0)


def test_a_paired_interval_is_tighter_than_the_difference_is_large() -> None:
    """Both orderings are measured on the same groups, so the interval belongs on the paired
    difference. Two orderings that agree everywhere have no uncertainty about their difference."""
    identical = np.ones(200, dtype=bool)
    interval = paired_interval(identical, identical, repeats=200)
    assert interval == pytest.approx({"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0})

    better = np.ones(200, dtype=bool)
    worse = np.zeros(200, dtype=bool)
    always = paired_interval(better, worse, repeats=200)
    assert always["delta"] == 1.0
    assert always["ci_low"] == 1.0


def test_a_hit_vector_answers_per_group() -> None:
    corpus = _corpus([4, 4], ["validation", "validation"])
    corpus.targets[:] = [9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0]
    scores = np.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.0])
    hits = hit_vector(corpus, scores, np.array([0, 1]), k=1)
    assert list(hits) == [True, False]


def test_every_reported_width_is_measured_on_its_own_groups() -> None:
    """A group of 20 candidates can be wrong at a shortlist of 8 and cannot be wrong at 48, so
    the widths do not share a denominator."""
    corpus = _corpus([12, 20, 60, 300], ["validation"] * 4, seed=4)
    shape = by_shortlist(corpus, np.random.default_rng(0).random(len(corpus.targets)), seed=1)
    assert shape["8"]["groups"] == 4
    assert shape["16"]["groups"] == 3
    assert shape["48"]["groups"] == 2
    for width in shape.values():
        assert 0.0 <= width["learned"] <= 1.0
        assert width["paired_vs_material"]["ci_low"] <= width["paired_vs_material"]["delta"]


def test_the_resample_count_is_the_one_the_protocol_fixed() -> None:
    """A preregistration names the number of resamples precisely because it is the kind of knob
    that looks free to turn after seeing an interval. The code follows the protocol."""
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "docs" / "prerank" / "protocol-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["uncertainty"]["repeats"] == BOOTSTRAP_REPEATS


def test_training_refuses_a_validation_split_it_cannot_measure_on() -> None:
    """Without this the first epoch's recall of zero beats the initial sentinel, becomes the best
    epoch, and the run reports a trained model whose validation evidence does not exist."""
    corpus = _corpus([60] * 10 + [10] * 5, ["train"] * 10 + ["validation"] * 5, seed=6)
    with pytest.raises(ValueError, match="no group larger than the shortlist"):
        train(corpus, Hyperparameters(max_epochs=1, k=48, seed=11))


def test_a_width_with_no_groups_is_reported_as_unmeasured() -> None:
    """The mean of no groups is a NaN, and a NaN in a results table is a number somebody will
    eventually read as a measurement."""
    corpus = _corpus([12, 20], ["validation", "validation"], seed=7)
    shape = by_shortlist(corpus, np.random.default_rng(0).random(len(corpus.targets)), seed=1)
    assert shape["48"] == {"groups": 0, "measured": False}
    assert shape["8"]["measured"] is True
    assert shape["8"]["groups"] == 2


def test_only_the_groups_the_orderings_disagree_about_count() -> None:
    """A paired comparison is decided by discordant pairs: groups both orderings get right, or
    both get wrong, say nothing about which is better."""
    candidate = np.array([True, True, False, True, False])
    reference = np.array([True, False, True, True, False])
    result = discordance(candidate, reference)
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["p_value"] == 1.0


def test_an_unambiguous_win_is_unambiguous() -> None:
    candidate = np.ones(30, dtype=bool)
    reference = np.zeros(30, dtype=bool)
    assert discordance(candidate, reference)["p_value"] < 1e-8


def test_two_orderings_that_never_disagree_have_nothing_to_test() -> None:
    same = np.array([True, False, True])
    assert discordance(same, same) == {"wins": 0, "losses": 0, "p_value": 1.0}


def test_the_exact_test_matches_hand_computed_tails() -> None:
    """Ten fair flips: five heads is the most likely outcome and cannot be evidence of anything;
    ten heads has probability 2 * 0.5**10."""
    assert _two_sided_binomial(5, 10) == pytest.approx(1.0)
    assert _two_sided_binomial(10, 10) == pytest.approx(2 * 0.5**10)
    assert _two_sided_binomial(9, 10) == pytest.approx(2 * (10 + 1) * 0.5**10)


def test_an_already_deployed_ordering_can_be_measured_on_the_same_footing(tmp_path) -> None:
    """The corpus carries the features the deployed value models consume, so a pre-ranker that is
    already in production can be scored here rather than approximated by a proxy for it."""
    from dicechess_training.prerank.export import export_ranker
    from dicechess_training.prerank.model import PreRankMLP
    from dicechess_training.prerank.train import onnx_scores

    corpus = _corpus([20, 30], ["validation", "validation"], seed=8)
    graph = export_ranker(PreRankMLP(9, [8]).double(), tmp_path / "model.onnx")
    scores = onnx_scores(corpus, graph)
    assert scores.shape == (len(corpus.targets),)
    assert np.isfinite(scores).all()
