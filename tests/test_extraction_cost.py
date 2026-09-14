"""Regression tests for measurement identity, coverage and fail-closed selection."""

import copy
import json

import pytest

from dicechess_training.ablation.extraction_cost import validate_extraction_cost
from dicechess_training.ablation.runner import (
    DEFAULT_PROTOCOL_PATH,
    _evaluate_extraction_cost_for_candidate,
    _load_extraction_cost,
)
from dicechess_training.contracts import kcp13


@pytest.fixture
def protocol():
    return json.loads(DEFAULT_PROTOCOL_PATH.read_bytes())


@pytest.fixture
def evidence(protocol):
    # Synthetic timings over the real public probe definitions, never benchmark results.
    return {
        "schema": "playground-extraction-benchmark-v1",
        "engine_version": protocol["engine_version"],
        "engine_artifact": f"com.fortemate:dicechess-engine_3:{protocol['engine_version']}",
        "runtime": dict.fromkeys(("java_version", "java_vendor", "os_name", "os_arch"), "test"),
        "config": {"warmup_iterations": 1, "sample_iterations": 2},
        "probes": {
            probe.id: {
                "fen": probe.fen,
                "schemas": {
                    key: {
                        "schema_id": schema["schema_id"],
                        "min_us": 90.0,
                        "median_us": 100.0,
                        "p95_us": 110.0,
                        "max_us": 120.0,
                    }
                    for key, schema in protocol["schemas"].items()
                },
            }
            for probe in kcp13.load_golden(kcp13.golden_path(protocol["engine_version"])).probes
        },
    }


def assert_rejected(evidence, protocol, tmp_path):
    with pytest.raises(ValueError):
        validate_extraction_cost(evidence, protocol)
    path = tmp_path / "cost.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        _load_extraction_cost(path, protocol)
    gate = _evaluate_extraction_cost_for_candidate("S1", evidence, protocol)
    assert gate["evidence_valid"] is False
    assert gate["cost_cleared"] is False


@pytest.mark.parametrize(
    "value", [-1, 0, float("nan"), float("inf"), -float("inf"), True, "100", None]
)
@pytest.mark.parametrize("schema", ["S0", "S1", "S2"])
def test_invalid_latency_fails_closed(evidence, protocol, tmp_path, schema, value):
    evidence["probes"]["start-w"]["schemas"][schema]["median_us"] = value
    assert_rejected(evidence, protocol, tmp_path)


@pytest.mark.parametrize(
    "defect", ["missing", "extra", "only-one", "fen", "schema", "schema-id", "quantiles"]
)
def test_incompatible_probe_evidence(evidence, protocol, tmp_path, defect):
    probes = evidence["probes"]
    if defect == "missing":
        del probes["passed-pawn-w"]
    elif defect == "extra":
        probes["unregistered"] = copy.deepcopy(probes["start-w"])
    elif defect == "only-one":
        evidence["probes"] = {"start-w": probes["start-w"]}
    elif defect == "fen":
        probes["start-w"]["fen"] = probes["start-b"]["fen"]
    elif defect == "schema":
        del probes["start-w"]["schemas"]["S2"]
    elif defect == "schema-id":
        probes["start-w"]["schemas"]["S1"]["schema_id"] = "kcp-13"
    else:
        probes["start-w"]["schemas"]["S1"]["min_us"] = 200.0
    assert_rejected(evidence, protocol, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime", {}),
        ("config", {}),
        ("probes", []),
        ("engine_artifact", "wrong"),
        ("engine_version", "wrong"),
        ("schema", "wrong"),
    ],
)
def test_invalid_provenance(evidence, protocol, tmp_path, field, value):
    evidence[field] = value
    assert_rejected(evidence, protocol, tmp_path)


def test_extraction_cost_threshold_and_missing_evidence(evidence, protocol):
    assert not _evaluate_extraction_cost_for_candidate("S1", None, protocol)["cost_cleared"]
    for median, passed in [(100.0, True), (110.0, True), (125.0, False)]:
        for probe in evidence["probes"].values():
            probe["schemas"]["S1"].update(median_us=median, p95_us=130.0, max_us=140.0)
        result = _evaluate_extraction_cost_for_candidate("S1", evidence, protocol)
        assert result["evidence_valid"] is True
        assert result["cost_cleared"] == passed


def test_valid_evidence_round_trip(evidence, protocol, tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(json.dumps(evidence))
    assert _load_extraction_cost(path, protocol) == evidence


def test_selection_cannot_bypass_cost_validation(evidence, protocol):
    import numpy as np

    from dicechess_training.ablation.runner import _evaluate_gate_for_candidate
    from dicechess_training.benchmark.metrics import scores

    y = np.ones(10)
    groups = np.arange(10)
    predictions = {"S0": [np.full(10, 0.5)] * 5, "S1": [np.full(10, 0.9)] * 5}
    summaries = {}
    for key, preds in predictions.items():
        metrics = scores(y, preds[0])
        summary = {f"{name}_mean": metrics[name] for name in ("log_loss", "brier", "ece")}
        summaries[key] = {
            "single_model_summary": summary,
            "single_model_unseen": summary,
            "single_model_slices": {},
        }
    invalid = copy.deepcopy(evidence)
    invalid["probes"]["start-w"]["schemas"]["S1"]["median_us"] = -1.0
    overbudget = copy.deepcopy(evidence)
    for probe in overbudget["probes"].values():
        probe["schemas"]["S1"].update(median_us=125.0, p95_us=130.0, max_us=140.0)
    for artifact, expected in [
        (evidence, True),
        (None, False),
        (invalid, False),
        (overbudget, False),
    ]:
        result = _evaluate_gate_for_candidate(
            "S1", summaries, predictions, y, groups, np.ones(10, dtype=bool), artifact, protocol
        )
        assert result["cleared"] == expected
