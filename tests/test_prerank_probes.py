"""The probe-pair gate, and the fixture it stands on.

Most of these tests are about the fixture rather than the gate, because a gate whose pairs have
quietly stopped being pairs passes everything and says nothing. Three of the first six pairs
written by hand were wrong; the certification is what caught them, so the certification is what
gets tested hardest.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dicechess_training.contracts import prerank as contract
from dicechess_training.prerank.export import export_ranker
from dicechess_training.prerank.model import PreRankMLP
from dicechess_training.prerank.probes import (
    BLUNDER,
    CERTIFICATION_MARGIN,
    SAFE,
    ProbePairError,
    certify,
    evaluate,
    load_pairs,
    pair_names,
    visible_differences,
)

ENGINE = "0.12.0"


def test_every_pair_has_both_halves() -> None:
    pairs = load_pairs(ENGINE)
    assert len(pairs) >= 6
    for pair in pairs.values():
        assert set(pair) == {BLUNDER, SAFE}


def test_every_pair_is_certified_by_the_engines_own_answer() -> None:
    """`queen_capture_danger` is the engine's dice-weighted "can this queen be taken next turn".
    A pair is a pair only when the blunder's is clearly higher."""
    margins = certify(ENGINE)
    assert set(margins) == set(pair_names(ENGINE))
    for name, margin in margins.items():
        assert margin >= CERTIFICATION_MARGIN, name


def test_the_halves_differ_only_in_the_opponents_placement() -> None:
    """Identical material and an identical placement of the mover's own pieces — otherwise the
    comparison is about something else and a pass means nothing."""
    columns = list(contract.columns(ENGINE))
    material = [columns.index(name) for name in ("p_diff", "n_diff", "b_diff", "r_diff", "q_diff")]
    material += [columns.index("material_diff"), columns.index("total_material")]
    for name, pair in load_pairs(ENGINE).items():
        blunder = pair[BLUNDER]["features"]
        safe = pair[SAFE]["features"]
        for index in material:
            assert blunder[index] == safe[index], f"{name} differs in {columns[index]}"


# This is the finding the gate exists to expose, so it is pinned rather than described. `kcp-13`'s
# last four columns are the capture probabilities and `rich-9` is exactly `kcp-13` without them,
# so the schema a pre-ranker can afford has no feature for a piece being en prise.
def test_rich9_sees_each_pair_through_mobility_alone() -> None:
    for name, difference in visible_differences(ENGINE).items():
        assert list(difference) == ["mobility_diff"], f"{name} differs in {list(difference)}"


def test_mobility_points_both_ways_across_the_fixture() -> None:
    """Which is why no rich-9 model can pass every pair by reading mobility: against pawn, knight
    and bishop attackers the safe twin has the higher mobility, and against rook attackers the
    lower. A fixture where they all agreed would be a gate that could be passed by accident."""
    directions = {
        name: difference["mobility_diff"][SAFE] > difference["mobility_diff"][BLUNDER]
        for name, difference in visible_differences(ENGINE).items()
    }
    assert any(directions.values()), "no pair where mobility favours the safe twin"
    assert not all(directions.values()), "no pair where mobility favours the blunder"


def test_a_missing_corpus_is_refused() -> None:
    with pytest.raises(ProbePairError, match="no committed probe-pair corpus"):
        certify("9.9.9")


def _artifact(tmp_path, weight: float):
    """A one-layer model whose only opinion is about mobility, with a chosen sign."""
    model = PreRankMLP(contract.feature_count(ENGINE), []).double()
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].weight[0, list(contract.columns(ENGINE)).index("mobility_diff")] = weight
        model.net[0].bias.zero_()
    return export_ranker(model, tmp_path / "model.onnx", ENGINE)


def test_a_ranker_that_likes_mobility_passes_exactly_the_pairs_where_mobility_agrees(
    tmp_path,
) -> None:
    """The gate's arithmetic, checked against a model whose answer is known in advance."""
    report = evaluate(_artifact(tmp_path, 1.0), ENGINE)
    favouring = {
        name
        for name, difference in visible_differences(ENGINE).items()
        if difference["mobility_diff"][SAFE] > difference["mobility_diff"][BLUNDER]
    }
    passed = {result["pair"] for result in report["results"] if result["passed"]}
    assert passed == favouring
    assert report["passed"] + report["failed"] == report["pairs"]


def test_reversing_the_sign_reverses_the_gate(tmp_path) -> None:
    liked = evaluate(_artifact(tmp_path / "a", 1.0), ENGINE)
    disliked = evaluate(_artifact(tmp_path / "b", -1.0), ENGINE)
    assert liked["passed"] + disliked["passed"] == liked["pairs"]


def test_a_ranker_with_no_opinion_fails_every_pair(tmp_path) -> None:
    """A tie is a failure. The seam keeps the top `k`, and a ranker that scores a blunder and the
    move that avoids it identically has not expressed a preference between them — which is
    exactly what the engine's material ordering does here."""
    report = evaluate(_artifact(tmp_path, 0.0), ENGINE)
    assert report["passed"] == 0
    assert all(result["margin"] == 0.0 for result in report["results"])


def test_the_material_ordering_cannot_separate_a_single_pair() -> None:
    """Not a model failing to learn something — the column is *identical* in both halves, so the
    pre-ranker the engine ships is exactly indifferent between a hanging queen and a safe one.
    That is the complaint this whole line of work started from."""
    columns = list(contract.columns(ENGINE))
    material = columns.index("material_diff")
    for name, pair in load_pairs(ENGINE).items():
        assert pair[BLUNDER]["features"][material] == pair[SAFE]["features"][material], name


def test_the_report_carries_what_a_reader_needs_to_judge_it(tmp_path) -> None:
    report = evaluate(_artifact(tmp_path, 1.0), ENGINE)
    assert report["engine_version"] == ENGINE
    for result in report["results"]:
        assert result["margin"] == pytest.approx(result["safe"] - result["blunder"])
        assert result["certified_danger_margin"] >= CERTIFICATION_MARGIN
        assert np.isfinite(result["safe"]) and np.isfinite(result["blunder"])
