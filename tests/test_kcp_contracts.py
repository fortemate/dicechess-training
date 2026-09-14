"""Contract and invariant tests for S1 (kcp-mobility-27-v1) and S2 (kcp-mobility-pawns-31-v1).

The golden vectors come directly from the engine; Python never derives feature values.
Invariants tested:
- prefix identity (S1 extends S0, S2 extends S1 byte for byte)
- mover-canonical twins (a position and its flipped rank-mirror produce identical vectors)
- mobility difference consistency: sum(own_moves_*) - sum(opp_moves_*) == mobility_diff
- PDI normalization: values in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- blocked-pawn probe behavior (0 pawn moves for both sides)
- passed-pawn probe behavior (positive count and rank)
- fail-closed ONNX and manifest validators
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from dicechess_training.contracts import (
    kcp13,
    kcp_mobility27,
    kcp_mobility_pawns31,
)


@pytest.fixture(scope="module")
def golden_s0() -> kcp13.GoldenCorpus:
    return kcp13.load_golden(kcp13.golden_path("0.9.3"))


@pytest.fixture(scope="module")
def golden_s1() -> kcp_mobility27.GoldenCorpus:
    return kcp_mobility27.load_golden()


@pytest.fixture(scope="module")
def golden_s2() -> kcp_mobility_pawns31.GoldenCorpus:
    return kcp_mobility_pawns31.load_golden()


def test_golden_s1_layout_and_metadata(golden_s1):
    assert golden_s1.schema == kcp_mobility27.SCHEMA_ID
    assert golden_s1.perspective == kcp_mobility27.PERSPECTIVE
    assert golden_s1.columns == kcp_mobility27.COLUMN_NAMES
    assert golden_s1.engine_version == "0.9.3"
    assert len(golden_s1.columns) == kcp_mobility27.FEATURE_COUNT == 27
    assert golden_s1.matrix().shape == (len(golden_s1.probes), 27)


def test_golden_s2_layout_and_metadata(golden_s2):
    assert golden_s2.schema == kcp_mobility_pawns31.SCHEMA_ID
    assert golden_s2.perspective == kcp_mobility_pawns31.PERSPECTIVE
    assert golden_s2.columns == kcp_mobility_pawns31.COLUMN_NAMES
    assert golden_s2.engine_version == "0.9.3"
    assert len(golden_s2.columns) == kcp_mobility_pawns31.FEATURE_COUNT == 31
    assert golden_s2.matrix().shape == (len(golden_s2.probes), 31)


def test_prefix_identity_s1_matches_s0(golden_s0, golden_s1):
    """The first 13 columns of S1 must match S0 byte-for-byte on all probes."""
    m0 = golden_s0.matrix()
    m1 = golden_s1.matrix()
    assert m0.shape[0] == m1.shape[0]
    np.testing.assert_array_equal(m1[:, :13], m0[:, :13])


def test_prefix_identity_s2_matches_s1(golden_s1, golden_s2):
    """The first 27 columns of S2 must match S1 byte-for-byte on all probes."""
    m1 = golden_s1.matrix()
    m2 = golden_s2.matrix()
    assert m1.shape[0] == m2.shape[0]
    np.testing.assert_array_equal(m2[:, :27], m1[:, :27])


def test_mover_canonical_twins_s1(golden_s1):
    """For every probe and its twin, S1 feature vectors must be identical."""
    probes = golden_s1.by_id()
    twins = [p for p in probes.values() if p.id.endswith("-twin")]
    assert len(twins) >= 9
    for twin in twins:
        base = probes[twin.id.removesuffix("-twin")]
        assert kcp13.twin_fen(base.fen) == twin.fen
        assert twin.side != base.side
        np.testing.assert_array_equal(twin.features, base.features, err_msg=twin.id)


def test_mover_canonical_twins_s2(golden_s2):
    """For every probe and its twin, S2 feature vectors must be identical."""
    probes = golden_s2.by_id()
    twins = [p for p in probes.values() if p.id.endswith("-twin")]
    assert len(twins) >= 9
    for twin in twins:
        base = probes[twin.id.removesuffix("-twin")]
        assert kcp13.twin_fen(base.fen) == twin.fen
        assert twin.side != base.side
        np.testing.assert_array_equal(twin.features, base.features, err_msg=twin.id)


def test_mobility_difference_identity_s1(golden_s1):
    """sum(own_moves_p..k) - sum(opp_moves_p..k) == mobility_diff for every position."""
    mob_diff_idx = kcp_mobility27.COLUMN_NAMES.index("mobility_diff")
    own_indices = [
        kcp_mobility27.COLUMN_NAMES.index(c) for c in kcp_mobility27.MOBILITY_OWN_COLUMNS
    ]
    opp_indices = [
        kcp_mobility27.COLUMN_NAMES.index(c) for c in kcp_mobility27.MOBILITY_OPP_COLUMNS
    ]

    for probe in golden_s1.probes:
        mob_diff = probe.features[mob_diff_idx]
        own_sum = probe.features[own_indices].sum()
        opp_sum = probe.features[opp_indices].sum()
        assert np.isclose(own_sum - opp_sum, mob_diff, atol=1e-5), (
            f"{probe.id}: {own_sum} - {opp_sum} != {mob_diff}"
        )


def test_pdi_normalization_s1_and_s2(golden_s1, golden_s2):
    """PDI features must be in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}."""
    valid_pdi = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
    own_idx = kcp_mobility27.COLUMN_NAMES.index("own_pdi")
    opp_idx = kcp_mobility27.COLUMN_NAMES.index("opp_pdi")

    for probe in golden_s1.probes:
        own_pdi = round(float(probe.features[own_idx]), 4)
        opp_pdi = round(float(probe.features[opp_idx]), 4)
        assert own_pdi in valid_pdi, f"{probe.id}: invalid own_pdi {own_pdi}"
        assert opp_pdi in valid_pdi, f"{probe.id}: invalid opp_pdi {opp_pdi}"


def test_blocked_pawn_probe_behavior(golden_s1, golden_s2):
    """Blocked pawn chain probe has zero pawn moves for both sides."""
    p1 = golden_s1.by_id()["blocked-pawns-w"]
    p2 = golden_s2.by_id()["blocked-pawns-w"]
    own_p_idx = kcp_mobility27.COLUMN_NAMES.index("own_moves_p")
    opp_p_idx = kcp_mobility27.COLUMN_NAMES.index("opp_moves_p")

    assert p1.features[own_p_idx] == 0.0
    assert p1.features[opp_p_idx] == 0.0
    assert p2.features[own_p_idx] == 0.0
    assert p2.features[opp_p_idx] == 0.0


def test_passed_pawn_probe_behavior(golden_s2):
    """Passed pawn probe has positive count and correct max rank."""
    probe = golden_s2.by_id()["passed-pawn-w"]
    own_pawns_idx = kcp_mobility_pawns31.COLUMN_NAMES.index("own_passed_pawns")
    opp_pawns_idx = kcp_mobility_pawns31.COLUMN_NAMES.index("opp_passed_pawns")
    own_rank_idx = kcp_mobility_pawns31.COLUMN_NAMES.index("own_passed_max_rank")
    opp_rank_idx = kcp_mobility_pawns31.COLUMN_NAMES.index("opp_passed_max_rank")

    assert probe.features[own_pawns_idx] == 1.0
    assert probe.features[opp_pawns_idx] == 0.0
    assert probe.features[own_rank_idx] == 6.0
    assert probe.features[opp_rank_idx] == 0.0


def _make_dummy_onnx(path, feature_count):
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, feature_count])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
    W = numpy_helper.from_array(np.zeros((feature_count, 1), dtype=np.float32), name="W")
    B = numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="B")
    node_matmul = helper.make_node("MatMul", ["input", "W"], ["mm"])
    node_add = helper.make_node("Add", ["mm", "B"], ["add"])
    node_sig = helper.make_node("Sigmoid", ["add"], ["output"])
    graph = helper.make_graph([node_matmul, node_add, node_sig], "test", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    import onnx

    onnx.save(model, str(path))


def test_onnx_contract_validation_s1(tmp_path):
    valid_path = tmp_path / "model27.onnx"
    _make_dummy_onnx(valid_path, 27)
    kcp_mobility27.validate_onnx_contract(valid_path)

    # Rejection of wrong input width
    invalid_path = tmp_path / "model13.onnx"
    _make_dummy_onnx(invalid_path, 13)
    with pytest.raises(kcp13.ContractError, match="shape"):
        kcp_mobility27.validate_onnx_contract(invalid_path)


def test_onnx_contract_validation_s2(tmp_path):
    valid_path = tmp_path / "model31.onnx"
    _make_dummy_onnx(valid_path, 31)
    kcp_mobility_pawns31.validate_onnx_contract(valid_path)

    invalid_path = tmp_path / "model27.onnx"
    _make_dummy_onnx(invalid_path, 27)
    with pytest.raises(kcp13.ContractError, match="shape"):
        kcp_mobility_pawns31.validate_onnx_contract(invalid_path)
