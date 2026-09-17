"""Validity guards of protocol v4: the no-information floor, the train-only scaler, and the
budget selected away from the validation partition (Issue #27)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dicechess_training.ablation import export
from dicechess_training.ablation.runner import (
    ValueMLP,
    _admissibility_for_schema,
    _fit_epochs,
    _select_schema,
    no_information_reference,
    predict,
    select_epoch_budget,
    standardisation_statistics,
    train_model,
)
from dicechess_training.contracts import SCHEMA_CONTRACTS

ROOT = Path(__file__).resolve().parents[1]
V4 = json.loads((ROOT / "docs/ablation/protocol-v4.json").read_text())
BASE_CONFIG = {
    "hidden_dims": [8],
    "learning_rate": 1e-3,
    "batch_size": 16,
    "epochs": 2,
    "loss": "bce-with-logits",
}


def _summary(log_loss: float, brier: float = 0.2, ece: float = 0.05) -> dict:
    return {"log_loss_mean": log_loss, "brier_mean": brier, "ece_mean": ece}


def _gate(cleared: bool) -> dict:
    return {"cleared": cleared, "checks": {}}


# --- the floor -------------------------------------------------------------------------------


def test_no_information_reference_scores_the_constant_base_rate_predictor():
    y_train = np.array([1.0] * 6 + [0.0] * 4)
    y_val = np.array([1.0, 0.0, 1.0, 0.0])
    reference = no_information_reference(y_train, y_val, np.array([True, False, True, False]), V4)
    assert reference["prediction"] == pytest.approx(0.6)
    expected = -np.mean(np.log([0.6, 0.4, 0.6, 0.4]))
    assert reference["scores"]["log_loss"] == pytest.approx(expected)
    assert reference["unseen_scores"]["count"] == 2


def test_protocols_without_an_admissibility_block_keep_their_behaviour():
    v3 = json.loads((ROOT / "docs/ablation/protocol-v3.json").read_text())
    assert "admissibility" not in v3
    assert (
        no_information_reference(np.array([1.0, 0.0]), np.array([1.0]), np.array([False]), v3)
        is None
    )
    assert _admissibility_for_schema(_summary(9.9), None) is None


@pytest.mark.parametrize("unsupported", [{"reference": "oracle"}, {"metric": "accuracy"}])
def test_unsupported_admissibility_configuration_fails_closed(unsupported):
    protocol = {"admissibility": {**V4["admissibility"], **unsupported}}
    with pytest.raises(ValueError, match="unsupported admissibility"):
        no_information_reference(np.array([1.0, 0.0]), np.array([1.0]), np.array([False]), protocol)


def test_a_candidate_worse_than_the_constant_is_inadmissible():
    reference = no_information_reference(
        np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([False, False]), V4
    )
    floor = reference["scores"]["log_loss"]
    assert _admissibility_for_schema(_summary(floor - 0.01), reference)["admissible"]
    assert not _admissibility_for_schema(_summary(floor + 0.01), reference)["admissible"]
    assert not _admissibility_for_schema(_summary(floor), reference)["admissible"]


def test_selection_refuses_an_inadmissible_baseline_and_names_both_losses():
    reference = no_information_reference(
        np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([False, False]), V4
    )
    floor = reference["scores"]["log_loss"]
    gates = {"S1": _gate(False), "S2": _gate(False)}
    inadmissible = {"S0": {"single_model_summary": _summary(floor + 0.5)}}
    assert _select_schema(gates, inadmissible, reference) is None

    admissible = {"S0": {"single_model_summary": _summary(floor - 0.5)}}
    assert _select_schema(gates, admissible, reference) == "S0"

    verdict = _admissibility_for_schema(inadmissible["S0"]["single_model_summary"], reference)
    assert verdict["candidate"] == pytest.approx(floor + 0.5)
    assert verdict["reference"] == pytest.approx(floor)


def test_a_cleared_candidate_still_wins_when_the_baseline_is_admissible():
    reference = no_information_reference(
        np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([False, False]), V4
    )
    results = {
        "S0": {"single_model_summary": _summary(0.60)},
        "S1": {"single_model_summary": _summary(0.55)},
        "S2": {"single_model_summary": _summary(0.50)},
    }
    assert _select_schema({"S1": _gate(True), "S2": _gate(False)}, results, reference) == "S1"
    assert _select_schema({"S1": _gate(True), "S2": _gate(True)}, results, reference) == "S2"


# --- the scaler ------------------------------------------------------------------------------


def test_standardisation_statistics_come_from_the_training_partition_only():
    x_train = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]], dtype=np.float32)
    x_val = np.full((3, 2), 1000.0, dtype=np.float32)  # a different distribution entirely
    mean, scale = standardisation_statistics(x_train, "train-statistics")
    assert mean == pytest.approx([2.0, 5.0])
    assert scale[0] == pytest.approx(np.std([0.0, 2.0, 4.0]))
    assert scale[1] == 1.0  # a constant column is centred, never amplified
    mean_all, _ = standardisation_statistics(np.vstack([x_train, x_val]), "train-statistics")
    assert mean_all[0] != pytest.approx(mean[0])


def test_the_model_carries_the_scaler_and_consumes_raw_features():
    x = np.array([[10.0, 100.0], [20.0, 300.0], [30.0, 200.0]], dtype=np.float32)
    y = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    config = {**BASE_CONFIG, "feature_standardisation": "train-statistics"}
    model = train_model(x, y, 2, config, 11)
    mean, scale = standardisation_statistics(x, "train-statistics")

    assert model.standardised
    assert set(model.state_dict()) >= {"feature_mean", "feature_scale"}
    assert model.state_dict()["feature_mean"].numpy() == pytest.approx(mean)
    # Raw features in, and the affine transform is the model's own first operation.
    manual = model.net(torch.tensor((x - mean) / scale, dtype=torch.float32)).detach().numpy()
    assert predict(model, x) == pytest.approx(manual.reshape(-1), abs=1e-6)


def test_an_unstandardised_model_keeps_the_historical_checkpoint_layout():
    model = train_model(
        np.ones((4, 2), np.float32), np.zeros(4, np.float32), 2, dict(BASE_CONFIG), 7
    )
    assert not model.standardised
    assert set(model.state_dict()) == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}


@pytest.mark.parametrize("mode", ["standardise", "zscore", 5])
def test_unknown_standardisation_fails_closed(mode):
    with pytest.raises(ValueError, match="unsupported feature standardisation"):
        standardisation_statistics(np.ones((2, 2), np.float32), mode)


def test_mismatched_standardisation_statistics_fail_closed():
    with pytest.raises(ValueError, match="together"):
        ValueMLP(2, [4], np.zeros(2), None)
    with pytest.raises(ValueError, match="shape"):
        ValueMLP(2, [4], np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        ValueMLP(2, [4], np.zeros(2), np.zeros(2))


# --- the budget ------------------------------------------------------------------------------


def _budget_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    x_fit = rng.normal(size=(96, 3)).astype(np.float32)
    y_fit = (x_fit[:, 0] > 0).astype(np.float32)
    x_inner = rng.normal(size=(48, 3)).astype(np.float32)
    y_inner = (x_inner[:, 0] > 0).astype(np.float32)
    return x_fit, y_fit, x_inner, y_inner


def test_budget_selection_scores_every_candidate_and_is_deterministic():
    x_fit, y_fit, x_inner, y_inner = _budget_inputs()
    config = {**BASE_CONFIG, "epoch_selection": {"candidates": [1, 2, 3]}}
    first = select_epoch_budget(x_fit, y_fit, x_inner, y_inner, 3, config, 11)
    second = select_epoch_budget(x_fit, y_fit, x_inner, y_inner, 3, config, 11)

    assert [c["epochs"] for c in first["candidates"]] == [1, 2, 3]
    assert (
        first["selected_epochs"]
        == min(first["candidates"], key=lambda c: (c["inner_log_loss"], c["epochs"]))["epochs"]
    )
    assert first == second


def test_budget_selection_rejects_an_empty_inner_split_and_out_of_range_candidates():
    x_fit, y_fit, x_inner, y_inner = _budget_inputs()
    config = {**BASE_CONFIG, "epoch_selection": {"candidates": [1]}}
    with pytest.raises(ValueError, match="inner tuning split"):
        select_epoch_budget(x_fit, y_fit, x_inner[:0], y_inner[:0], 3, config, 11)

    bad = {**BASE_CONFIG, "epoch_selection": {"candidates": [0, 200]}}
    with pytest.raises(ValueError, match="candidates"):
        select_epoch_budget(x_fit, y_fit, x_inner, y_inner, 3, bad, 11)


def test_checkpointed_training_matches_training_straight_to_that_budget():
    x_fit, y_fit, _, _ = _budget_inputs(3)
    config = {**BASE_CONFIG, "feature_standardisation": "train-statistics"}
    direct = train_model(x_fit, y_fit, 3, config, 23, epochs=3)
    checkpointed = None
    for epoch, model in _fit_epochs(x_fit, y_fit, 3, config, 23, 5, frozenset({3, 5})):
        if epoch == 3:
            checkpointed = predict(model, x_fit)
            break
    assert checkpointed == pytest.approx(predict(direct, x_fit), abs=1e-6)


# --- the exported graph ----------------------------------------------------------------------


def test_the_exported_graph_takes_raw_features_and_matches_the_pipeline(tmp_path):
    rng = np.random.default_rng(5)
    contract = SCHEMA_CONTRACTS["kcp-13"]
    x = rng.normal(4.0, 6.0, size=(64, contract.FEATURE_COUNT)).astype(np.float32)
    y = (rng.random(64) < 0.5).astype(np.float32)
    config = {**BASE_CONFIG, "feature_standardisation": "train-statistics"}
    model = train_model(x, y, contract.FEATURE_COUNT, config, 11)

    path = export.export_candidate(model, "kcp-13", tmp_path / "candidate.onnx")
    contract.validate_onnx_contract(path)  # ADR 0001 names, shapes, dtype and opset ceiling
    assert export.probe_parity(model, "kcp-13", path) < 1e-5

    # The graph must be fed raw features: pre-standardised input would double-apply the transform.
    features = export.golden_probe_features("kcp-13")
    mean, scale = standardisation_statistics(x, "train-statistics")
    assert (
        np.abs(contract.predict(path, (features - mean) / scale) - predict(model, features)).max()
        > 1e-6
    )


# --- shard provenance ------------------------------------------------------------------------


def test_content_digest_ignores_footer_layout_but_not_data_or_semantics(tmp_path):
    """A re-run must match by content even though its file bytes differ (#27)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dicechess_training.schema import shard_content_digest

    metadata = {
        b"dicechess_training_schema": b"v0-enriched",
        b"feature_schema": b"kcp-13",
        b"engine_version": b"0.9.3",
        b"ruleset": b"standard-dicechess-v1",
        b"perspective": b"side-to-move",
    }
    table = pa.table(
        {"game_id": ["b", "a"], "ply": [1, 2], "value": [1.5, 2.5]},
        metadata=metadata,
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(table, first, compression="snappy")
    # Same rows, different physical layout: more row groups and a different encoding choice.
    pq.write_table(table, second, compression="gzip", row_group_size=1, use_dictionary=False)
    assert first.read_bytes() != second.read_bytes()
    assert shard_content_digest(first) == shard_content_digest(second)

    changed_value = tmp_path / "changed-value.parquet"
    pq.write_table(
        pa.table({"game_id": ["b", "a"], "ply": [1, 2], "value": [1.5, 9.5]}, metadata=metadata),
        changed_value,
    )
    assert shard_content_digest(changed_value) != shard_content_digest(first)

    changed_engine = tmp_path / "changed-engine.parquet"
    pq.write_table(
        table.replace_schema_metadata({**metadata, b"engine_version": b"0.9.4"}), changed_engine
    )
    assert shard_content_digest(changed_engine) != shard_content_digest(first)


# --- rendering a run that selected nothing ---------------------------------------------------


def _minimal_report(selected: str | None, ci_by_schema: dict[str, list[float] | None]) -> dict:
    schemas = {
        key: {
            "schema_id": sid,
            "feature_count": count,
            "single_model_summary": {
                "log_loss_mean": 0.6,
                "log_loss_std": 0.01,
                "brier_mean": 0.22,
                "brier_std": 0.001,
                "ece_mean": 0.03,
                "ece_std": 0.002,
            },
        }
        for key, sid, count in (
            ("S0", "kcp-13", 13),
            ("S1", "kcp-mobility-27-v1", 27),
            ("S2", "kcp-mobility-pawns-31-v1", 31),
        )
    }
    gates = {
        key: {
            "cleared": False,
            "log_loss_delta_ci_95": ci_by_schema[key],
            "relative_log_loss_gain": -0.01,
            "checks": {},
        }
        for key in ("S1", "S2")
    }
    decision = {
        "selected_schema": selected,
        "selected_schema_id": None if selected is None else schemas[selected]["schema_id"],
        "gate_results": {"S1": False, "S2": False},
    }
    if selected is None:
        decision["inadmissible_reason"] = "the baseline S0 scored worse than the reference"
    return {
        "protocol_version": "test",
        "protocol_sha256": "0" * 64,
        "engine_version": "0.9.3",
        "schemas": schemas,
        "gate_evaluations": gates,
        "decision": decision,
        "split_summary": {
            "total_positions": 10,
            "decisive_positions": 10,
            "train_positions": 8,
            "val_positions": 1,
            "test_positions": 1,
            "train_games": 8,
            "val_games": 1,
            "test_games": 1,
        },
    }


@pytest.mark.parametrize("selected", [None, "S0", "S1"])
def test_the_header_verdict_matches_the_decision(selected):
    from dicechess_training.ablation.report import _render_header

    report = _minimal_report(selected, {"S1": [-0.01, 0.02], "S2": [-0.01, 0.02]})
    rendered = "\n".join(_render_header(report, report["split_summary"], report["decision"]))
    if selected is None:
        assert "no schema was selected" in rendered
        assert "the baseline S0 scored worse than the reference" in rendered
        assert "cleared all gate rules" not in rendered
        assert "None" not in rendered
    elif selected == "S0":
        assert "Neither wider candidate cleared" in rendered
        assert "no schema was selected" not in rendered
    else:
        assert "cleared all gate rules" in rendered
        assert "kcp-mobility-27-v1" in rendered


@pytest.mark.parametrize(
    "ci,expected",
    [
        ([-0.05, -0.01], "entirely below zero"),
        ([0.01, 0.05], "entirely above zero"),
        ([-0.01, 0.05], "spans zero"),
        (None, "unavailable"),
    ],
)
def test_the_interval_reading_follows_the_interval(ci, expected):
    from dicechess_training.ablation.report import _render_interpretation

    report = _minimal_report("S0", {"S1": ci, "S2": ci})
    lines = _render_interpretation(
        report["schemas"], report["gate_evaluations"], report["split_summary"]
    )
    rendered = "\n".join(lines)
    assert expected in rendered
    if ci is None:
        assert "spans zero" not in rendered
