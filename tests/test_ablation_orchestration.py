"""Orchestration contracts on authored predictions, never measured model results."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dicechess_training.ablation import runner
from dicechess_training.contracts import SCHEMA_CONTRACTS

GOLDENS = Path(__file__).parent / "fixtures/ablation-orchestration.json"


def _plain(value):
    if isinstance(value, np.ndarray):
        return ["array", value.dtype.str, list(value.shape), value.tolist()]
    if isinstance(value, dict):
        return ["dict", [[key, _plain(item)] for key, item in value.items()]]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_plain(item) for item in value]]
    return value


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            _plain(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _single_arguments(sid, unseen, budget):
    columns = list(SCHEMA_CONTRACTS[sid].COLUMN_NAMES)
    frame = pd.DataFrame(
        {column: np.arange(6, dtype=np.float32) + index / 4 for index, column in enumerate(columns)}
    )
    fitting = np.array([True, True, True, True, False, False])
    validating = ~fitting
    model = {"hidden_dims": [2], "learning_rate": 0.001, "batch_size": 2, "epochs": 2}
    if budget != "fixed":
        model["epoch_selection"] = {"method": "authored-budget"}
        model["feature_standardisation"] = "train-statistics"
    protocol = {"model": model, "seeds": [2, 4]}
    inner = np.array([False, True, False, True]) if budget == "selected" else None
    return (
        "authored-schema",
        sid,
        frame,
        fitting,
        validating,
        frame.loc[validating].reset_index(drop=True),
        np.array([0, 1, 0, 1], dtype=np.float32),
        np.array([0, 1], dtype=np.float64),
        np.array([unseen, False]),
        protocol,
        inner,
    )


def _authored_scores(y, prediction):
    """Arithmetic definitions exercise aggregation without claiming evaluation quality."""
    return {
        "log_loss": float(prediction.sum()),
        "brier": float((prediction * prediction).sum()),
        "ece": len(y) / 8,
        "count": len(y),
    }


def _single_case(monkeypatch, sid, unseen, budget):
    arguments = _single_arguments(sid, unseen, budget)
    original_frame = arguments[2].copy(deep=True)
    original_protocol = copy.deepcopy(arguments[9])
    events = []

    def select(x_fit, y_fit, x_inner, y_inner, dimension, config, seed):
        events.append(
            ["budget", seed, dimension, x_fit.copy(), y_fit.copy(), x_inner.copy(), y_inner.copy()]
        )
        assert config is arguments[9]["model"]
        return {"selected_epochs": seed // 2, "inner_log_loss": seed / 8}

    def train(x, y, dimension, config, seed, epochs):
        assert config is arguments[9]["model"]
        events.append(["train", seed, epochs, dimension, x.copy(), y.copy()])
        return seed

    def predict(model, x):
        events.append(["predict", model, x.copy()])
        return np.array([model / 8, model / 8 + 0.25], dtype=np.float64)

    def probes(model, schema, protocol):
        assert protocol is arguments[9]
        events.append(["probes", model, schema])
        return {"authored-a": model / 8, "authored-b": model / 8 + 0.25}

    def score(y, p):
        events.append(["scores", y.copy(), p.copy()])
        return _authored_scores(y, p)

    def slices(frame, y, p):
        pd.testing.assert_frame_equal(frame, arguments[5], check_exact=True)
        events.append(["slices", y.copy(), p.copy()])
        return {"authored-slice": _authored_scores(y, p)}

    def evaluate(predictions, protocol):
        assert protocol is arguments[9]
        events.append(["evaluate-probes", dict(predictions)])
        return {"checks": {"authored": True}, "predictions": dict(predictions)}

    monkeypatch.setattr(runner, "select_epoch_budget", select)
    monkeypatch.setattr(runner, "train_model", train)
    monkeypatch.setattr(runner, "predict", predict)
    monkeypatch.setattr(runner, "predict_probes", probes)
    monkeypatch.setattr(runner, "scores", score)
    monkeypatch.setattr(runner, "compute_slices", slices)
    monkeypatch.setattr(runner, "evaluate_probe_predictions", evaluate)
    result = runner._train_single_schema(*arguments)
    pd.testing.assert_frame_equal(arguments[2], original_frame, check_exact=True)
    assert arguments[9] == original_protocol
    assert [event[1] for event in events if event[0] == "train"] == [2, 4]
    assert len(result[2]) == 2
    assert result[0]["single_model_unseen"]["count"] == int(unseen)
    return {"result": result, "events": events}


@pytest.mark.parametrize("sid", list(SCHEMA_CONTRACTS))
@pytest.mark.parametrize("unseen", [False, True])
@pytest.mark.parametrize("budget", ["fixed", "selected", "requested-without-mask"])
def test_complete_schema_outputs_training_order_and_aggregation(
    monkeypatch, capsys, sid, unseen, budget
):
    value = _single_case(monkeypatch, sid, unseen, budget)
    value["stdout"] = capsys.readouterr().out
    expected = json.loads(GOLDENS.read_text())[f"single:{sid}:{int(unseen)}:{budget}"]
    assert _fingerprint(value) == expected


def test_invalid_inner_mask_fails_before_training_or_output(monkeypatch, capsys):
    arguments = list(_single_arguments("kcp-13", True, "selected"))
    arguments[-1] = np.array([True])

    def unexpected(*args, **kwargs):
        raise AssertionError("training must not start")

    monkeypatch.setattr(runner, "train_model", unexpected)
    with pytest.raises(ValueError, match="inner tuning mask must align"):
        runner._train_single_schema(*arguments)
    assert capsys.readouterr().out == ""


def _run_case(monkeypatch, root, metric, selected, defaults, cost_present=True):
    root.mkdir()
    order = ["S2", "S0", "S1"]
    schemas = dict(zip(("S0", "S1", "S2"), SCHEMA_CONTRACTS, strict=True))
    protocol = {
        "protocol_version": "authored-orchestration",
        "engine_version": "0.9.3",
        "schemas": {key: {"schema_id": schemas[key]} for key in order},
    }
    protocol_file = root / "protocol.json"
    protocol_file.write_text(json.dumps(protocol))
    base = root / "data/enriched"
    for key in order:
        folder = base / schemas[key]
        folder.mkdir(parents=True)
        for name in ("b.parquet", "a.parquet"):
            (folder / name).write_bytes(f"authored:{key}:{name}".encode())
    cost = root / "tests/fixtures/benchmark/extraction-cost-0.9.3.json"
    if cost_present:
        cost.parent.mkdir(parents=True)
        cost.write_text("{}")
    events = []
    frames = {
        key: pd.DataFrame({"result": np.array([0, 1, 0, 1], dtype=np.float32)}) for key in order
    }
    fitting = np.array([True, True, False, False])
    validating = ~fitting
    val_frame = frames["S0"].loc[validating].reset_index(drop=True)
    y_val = np.array([0, 1], dtype=np.float64)
    unseen = np.array([True, False])
    inner = np.array([False, True])
    reference = (
        None
        if metric is None
        else {
            "reference": "authored-reference",
            "metric": metric,
            "prediction": 0.5,
            "scores": {metric: 0.5},
        }
    )
    results = {}
    predictions = {}
    for key in order:
        value = 0.75 if selected is None or key not in ("S0", selected) else 0.25
        results[key] = {
            "schema_id": schemas[key],
            "single_model_summary": {"log_loss_mean": value, "brier_mean": value},
        }
        predictions[key] = [np.array([0.25, 0.75])]
    original_admissibility = runner._admissibility_for_schema

    def load_cost(path, loaded):
        events.append("cost")
        assert loaded == protocol
        assert path == (cost if cost_present or not defaults else None)
        return {"authored-cost": True} if cost_present else None

    def read(folder, sid, version):
        key = next(key for key in order if schemas[key] == sid)
        events.append(f"read:{key}")
        assert folder == base / sid
        assert version == "0.9.3"
        return frames[key]

    def integrity(loaded):
        events.append("integrity")
        assert list(loaded) == order
        assert all(loaded[key] is frames[key] for key in order)

    def split(frame, loaded):
        events.append("splits")
        assert frame is frames["S0"]
        assert loaded == protocol
        return (
            fitting,
            validating,
            val_frame,
            y_val,
            np.array(["a", "b"]),
            {"authored-split": True},
            unseen,
            inner,
        )

    def train(
        key,
        sid,
        frame,
        train_mask,
        val_mask,
        validation,
        y_train,
        labels,
        unseen_mask,
        loaded,
        inner_mask,
    ):
        events.append(f"train:{key}")
        assert sid == schemas[key]
        assert frame is frames[key]
        assert train_mask is fitting
        assert val_mask is validating
        assert validation is val_frame
        assert labels is y_val
        np.testing.assert_array_equal(y_train, [0, 1])
        assert unseen_mask is unseen
        assert inner_mask is inner
        assert loaded == protocol
        return results[key], np.array([0.5, 0.5]), predictions[key]

    def no_information(y_train, labels, mask, loaded):
        events.append("reference")
        np.testing.assert_array_equal(y_train, [0, 1])
        assert labels is y_val
        assert mask is unseen
        assert loaded == protocol
        return reference

    def admissibility(summary, supplied):
        key = next(key for key in order if results[key]["single_model_summary"] is summary)
        events.append(f"admissibility:{key}")
        assert supplied is reference
        return original_admissibility(summary, supplied)

    def gate(
        key,
        supplied,
        seed_predictions,
        labels,
        groups,
        mask,
        loaded_cost,
        loaded,
        supplied_reference,
    ):
        events.append(f"gate:{key}")
        for k in order:
            assert supplied[k] is results[k]
            assert seed_predictions[k] is predictions[k]
        assert labels is y_val
        assert mask is unseen
        assert loaded == protocol
        assert supplied_reference is reference
        assert loaded_cost == ({"authored-cost": True} if cost_present else None)
        return {"cleared": key == selected, "authored-gate": key}

    def select(gates, supplied, supplied_reference):
        events.append("select")
        assert list(gates) == ["S1", "S2"]
        assert all(supplied[k] is results[k] for k in order)
        assert supplied_reference is reference
        return selected

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "DEFAULT_PROTOCOL_PATH", protocol_file)
    monkeypatch.setattr(runner, "_load_extraction_cost", load_cost)
    monkeypatch.setattr(runner, "read_enriched_shards", read)
    monkeypatch.setattr(
        runner, "shard_content_digest", lambda path: "authored-content:" + path.name
    )
    monkeypatch.setattr(runner, "_validate_cross_schema_integrity", integrity)
    monkeypatch.setattr(runner, "_prepare_dataset_splits", split)
    monkeypatch.setattr(runner, "_train_single_schema", train)
    monkeypatch.setattr(runner, "no_information_reference", no_information)
    monkeypatch.setattr(runner, "_admissibility_for_schema", admissibility)
    monkeypatch.setattr(runner, "_evaluate_gate_for_candidate", gate)
    monkeypatch.setattr(runner, "_select_schema", select)
    report = runner.run_ablation() if defaults else runner.run_ablation(protocol_file, base, cost)
    assert all(report["schemas"][key] is results[key] for key in order)
    assert report["no_information_reference"] is reference
    assert list(report["schemas"]) == order
    expected = [
        "cost",
        *[f"read:{key}" for key in order],
        "integrity",
        "splits",
        *[f"train:{key}" for key in order],
        "reference",
    ]
    if reference is not None:
        expected += [f"admissibility:{key}" for key in order]
    assert events == [*expected, "gate:S1", "gate:S2", "select"]
    return report


RUN_CASES = [(None, key) for key in ("S0", "S1", "S2")] + [
    (metric, key) for metric in ("log_loss", "brier") for key in (None, "S0", "S1", "S2")
]


@pytest.mark.parametrize(("metric", "selected"), RUN_CASES)
@pytest.mark.parametrize("defaults", [False, True])
def test_complete_ablation_report_admissibility_decision_and_call_order(
    monkeypatch, tmp_path, capsys, metric, selected, defaults
):
    root = tmp_path / "project"
    report = _run_case(monkeypatch, root, metric, selected, defaults)
    text = capsys.readouterr().out
    expected_prefix = (
        f"Loaded protocol: authored-orchestration (SHA-256: {report['protocol_sha256'][:12]}...)\n"
    )
    for key in ("S2", "S0", "S1"):
        sid = report["schemas"][key]["schema_id"]
        expected_prefix += (
            f"Loading enriched shards for {key} ({sid}) from {root / 'data/enriched' / sid}...\n"
        )
    assert text.startswith(expected_prefix)
    suffix = text[len(expected_prefix) :]
    expected = json.loads(GOLDENS.read_text())[f"run:{metric}:{selected}"]
    assert _fingerprint({"report": report, "stdout_after_loading": suffix}) == expected


def test_missing_default_extraction_cost_stays_absent(monkeypatch, tmp_path, capsys):
    report = _run_case(monkeypatch, tmp_path / "project", None, "S0", True, cost_present=False)
    assert report["extraction_cost"] is None
    assert report["decision"]["selected_schema"] == "S0"
    assert "No-information reference" not in capsys.readouterr().out
