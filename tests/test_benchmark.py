"""Independent arithmetic, group leakage, identity tampering and qualification regressions."""

import copy
import json
import math
import shutil

import jsonschema
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from dicechess_training.benchmark import core
from dicechess_training.benchmark.__main__ import main
from dicechess_training.benchmark.metrics import confidence, scores
from dicechess_training.benchmark.splits import assignments, leakage, position_key, split_for
from dicechess_training.contracts import kcp13

FIXTURE = core.ROOT / "tests/fixtures/benchmark"


def write_json(path, value):
    path.write_text(json.dumps(value))


def copy_data(tmp_path, name="data", *, kind="synthetic", rename=False):
    target = tmp_path / name
    shutil.copytree(FIXTURE, target)
    manifest = core.read_json(target / "manifest.json")
    rows = core.read_json(target / "rows.json")
    if rename:
        for row in rows:
            row["group_id"] = "sealed-" + row["group_id"]
            row["game_id"] = "sealed-" + row["game_id"]
    manifest["kind"] = kind
    save_data(target, manifest, rows)
    return target


def save_data(directory, manifest, rows):
    write_json(directory / "rows.json", rows)
    manifest["rows_sha256"] = kcp13.sha256_of(directory / "rows.json")
    write_json(directory / "manifest.json", manifest)


def candidate(tmp_path, data_dir=FIXTURE, name="candidate", weight=0.1):
    directory = tmp_path / name
    directory.mkdir()
    # An untrained material sigmoid, generated only in temporary test storage.
    weights = np.zeros((13, 1), dtype=np.float32)
    weights[5] = weight
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weights"], ["logit"]),
            helper.make_node("Sigmoid", ["logit"], ["output"]),
        ],
        "synthetic-contract-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 13])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.save(model, directory / "model.onnx")
    data, rows = core.load_dataset(data_dir)
    provenance = {
        **core.train_identity(data, rows),
        "benchmark_sha256": core.digest(core.load_protocol()),
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "seed": 0,
        "engine_version": "0.9.2",
        "perspective": "side-to-move",
    }
    manifest = kcp13.build_manifest(
        directory / "model.onnx", "synthetic-test", ">=0.9.2 <0.10.0", provenance=provenance
    )
    write_json(directory / "manifest.json", manifest)
    return directory


def final_inputs(tmp_path):
    # All contents remain synthetic test fixtures; role flags only exercise the private validator.
    dev = copy_data(tmp_path, "dev", kind="public-licensed")
    final = copy_data(tmp_path, "final", kind="owner-controlled", rename=True)
    model = candidate(tmp_path, dev)
    seal = {
        "schema": "playground-seal-v1",
        "implementation_sha256": core.implementation_digest(),
        "benchmark_sha256": core.digest(core.load_protocol()),
        "candidate_manifest_sha256": core.digest(core.read_json(model / "manifest.json")),
        "development_dataset_sha256": core.digest(core.read_json(dev / "manifest.json")),
        "final_dataset_sha256": core.digest(core.read_json(final / "manifest.json")),
        "accepted_references": [],
        "promotion_reference": "no-information",
        "serving_limits": {"latency_p95_ms": 10, "rss_mb": 100},
    }
    path = tmp_path / "seal.json"
    write_json(path, seal)
    return dev, final, model, path


def run_final(dev, final, model, seal, **kwargs):
    return core.evaluate(
        final,
        model,
        mode="final",
        development_dir=dev,
        seal_path=seal,
        seal_sha256=kcp13.sha256_of(seal),
        **kwargs,
    )


def test_hand_calculated_binary_scores_and_bins():
    # Two correct predictions at .75/.25: loss -log(.75), Brier .25^2, ECE .25.
    result = scores([0, 1], [0.25, 0.75])
    assert result["log_loss"] == pytest.approx(-math.log(0.75))
    assert result["brier"] == 0.0625
    assert result["prevalence"] == 0.5
    assert result["ece"] == 0.25
    assert sum(b["count"] for b in result["calibration"]) == 2
    assert result["calibration"][0]["prediction"] is None
    edges = scores([0, 1, 1], [0, 0.1, 1])["calibration"]
    assert [edges[i]["count"] for i in (0, 1, 9)] == [1, 1, 1]
    assert np.isfinite(scores([1, 0], [0, 1])["log_loss"])


@pytest.mark.parametrize(
    "y,p",
    [([], []), ([1], [np.nan]), ([1], [1.1]), ([0.5], [0.5]), ([0, 1], [0.5]), ([[1]], [[0.5]])],
)
def test_invalid_scores(y, p):
    with pytest.raises(ValueError):
        scores(y, p)


def test_group_bootstrap_against_direct_resampling():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.9, 0.6])
    group_rows = [np.array([0, 1, 2]), np.array([3])]
    result = confidence(y, p, ["a", "a", "a", "b"], [0.5] * 4, repeats=100, seed=23)
    rng = np.random.default_rng(23)
    direct = []
    for _ in range(100):
        idx = np.concatenate([group_rows[i] for i in rng.integers(2, size=2)])
        s = scores(y[idx], p[idx])
        direct.append([s["log_loss"], s["brier"], s["ece"]])
    expected = np.quantile(direct, [0.025, 0.975], axis=0)
    for i, name in enumerate(("log_loss", "brier", "ece")):
        assert result["intervals"][name] == pytest.approx(expected[:, i])
    assert confidence([0, 1], [0.5, 0.5], ["a", "a"])["intervals"] is None
    paired = confidence(y, p, ["a", "a", "a", "b"], p, repeats=100)
    assert paired["intervals"]["log_loss_delta"] == [0, 0]
    assert paired["intervals"]["brier_delta"] == [0, 0]


def test_splits_stable_under_shuffle_extension_and_repeated_positions():
    _, rows = core.load_dataset(FIXTURE)
    parts = assignments(rows)
    by_id = {r["id"]: s for r, s in zip(rows, parts, strict=True)}
    shuffled = rows[::-1]
    assert all(by_id[r["id"]] == s for r, s in zip(shuffled, assignments(shuffled), strict=True))
    assert (
        assignments(rows + [{**rows[0], "id": "new", "group_id": "new", "game_id": "new"}])[:-1]
        == parts
    )
    assert set(parts) == {"train", "validation", "test"}
    assert leakage(rows, parts)["train:validation"] > 0
    duplicate = {**rows[0], "fen": " ".join(rows[0]["fen"].split()[:4]) + " 8 20", "dice": "KKK"}
    assert position_key(duplicate) == position_key(rows[0])


@pytest.mark.parametrize("field", ["game_id", "root_id"])
def test_group_alias_cannot_cross_splits(field):
    _, rows = core.load_dataset(FIXTURE)
    first, second = copy.deepcopy(rows[:2])
    first[field] = second[field] = "shared"
    with pytest.raises(ValueError, match="crosses"):
        assignments([first, second])


@pytest.mark.parametrize(
    "field,value",
    [
        ("license", "unknown"),
        ("kind", "unverified"),
        ("rows_sha256", "a" * 64),
        ("golden_sha256", "a" * 64),
        ("source_sha256", ""),
        ("license_evidence_sha256", "b" * 64),
        ("perspective", "white"),
        ("engine_version", "0.9.1"),
        ("feature_schema", "kcp-14"),
        ("version", "/private/secret"),
        ("target", "expected-score"),
    ],
)
def test_missing_or_invalid_dataset_provenance(tmp_path, field, value):
    directory = copy_data(tmp_path)
    manifest = core.read_json(directory / "manifest.json")
    manifest[field] = value
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        core.load_dataset(directory)


def test_missing_license_evidence_and_duplicate_json_keys(tmp_path):
    directory = copy_data(tmp_path)
    (directory / "license.txt").unlink()
    with pytest.raises(FileNotFoundError):
        core.load_dataset(directory)
    (directory / "manifest.json").write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="duplicate"):
        core.load_dataset(directory)


@pytest.mark.parametrize("mutation", ["features", "side", "duplicate", "outcome", "columns"])
def test_invalid_rows_fail_closed(tmp_path, mutation):
    directory = copy_data(tmp_path)
    manifest, rows = core.load_dataset(directory)
    if mutation == "features":
        rows[0]["features"][0] += 1
    elif mutation == "side":
        rows[0]["side"] = "b" if rows[0]["side"] == "w" else "w"
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "outcome":
        rows[0]["result"] = 0.2
    else:
        manifest["columns"] = manifest["columns"][::-1]
    save_data(directory, manifest, rows)
    with pytest.raises(ValueError):
        core.load_dataset(directory)


@pytest.fixture(scope="module")
def public_report():
    return core.evaluate(FIXTURE)


def test_public_report_schema_privacy_and_development_refusal(public_report):
    schema = core.read_json(core.ROOT / "docs/benchmark/report-schema-v1.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(public_report, schema)
    assert public_report["decision"]["status"] == "not-qualified"
    assert "development-only" in public_report["decision"]["reasons"]
    assert public_report["slices"]["target:uncertain"]["status"] == "unsupported"
    assert public_report["exact_unseen"] is None
    text = json.dumps(public_report)
    for private in (str(FIXTURE), "synthetic-game", "rnbqkbnr", "license.txt", "features"):
        assert private not in text
    broken = copy.deepcopy(public_report)
    del broken["aggregate"]["candidate"]["calibration"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)
    broken = copy.deepcopy(public_report)
    broken["private_path"] = "forbidden"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)


def test_no_test_metrics_or_test_tuning(tmp_path, public_report):
    directory = copy_data(tmp_path)
    manifest, rows = core.load_dataset(directory)
    for row in rows:
        if split_for(row["group_id"]) == "test":
            row["result"] = 1 - row["result"]
    save_data(directory, manifest, rows)
    changed = core.evaluate(directory)
    assert changed["aggregate"] == public_report["aggregate"]
    assert changed["baselines"] == public_report["baselines"]


def test_draw_exclusion_is_explicit(tmp_path):
    directory = copy_data(tmp_path)
    manifest, rows = core.load_dataset(directory)
    rows[0]["result"] = 0.5
    save_data(directory, manifest, rows)
    report = core.evaluate(directory)
    assert report["counts"]["excluded_draws"] == 1
    assert "conditioned-on-decisive" in report["target"]


def test_candidate_golden_and_provenance(tmp_path):
    directory = candidate(tmp_path)
    report = core.evaluate(FIXTURE, directory)
    assert all(report["probes"]["checks"].values())
    assert report["probes"]["king_capture"] == "requires-serving-evidence"
    manifest = core.read_json(directory / "manifest.json")
    del manifest["provenance"]["config_sha256"]
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(KeyError):
        core.evaluate(FIXTURE, directory)


@pytest.mark.parametrize(
    "field", ["training_data_sha256", "training_groups_sha256", "benchmark_sha256"]
)
def test_candidate_bound_to_training_and_protocol(tmp_path, field):
    directory = candidate(tmp_path)
    manifest = core.read_json(directory / "manifest.json")
    manifest["provenance"][field] = "f" * 64
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        core.evaluate(FIXTURE, directory)


def test_gate_hand_calculated_pass_and_regressions():
    protocol = core.load_protocol()
    summary = {
        "candidate": {"log_loss": 0.6, "brier": 0.2, "ece": 0.01},
        "reference": {"log_loss": 0.7, "brier": 0.25, "ece": 0.02},
        "confidence": {"groups": 50, "intervals": {"log_loss_delta": [-0.2, -0.05]}},
    }
    assert core.gate_check(summary, protocol) == []
    summary["candidate"]["log_loss"] = 0.8
    assert "insufficient-relative-gain" in core.gate_check(summary, protocol)
    assert "slice-log-loss-regression" in core.gate_check(summary, protocol, critical_slice=True)
    summary["confidence"]["groups"] = 1
    summary["confidence"]["intervals"] = None
    assert "insufficient-groups" in core.gate_check(summary, protocol)
    assert "uncertain-improvement" in core.gate_check(summary, protocol)


def test_sealed_final_leakage_and_missing_serving_evidence(tmp_path):
    args = final_inputs(tmp_path)
    result = run_final(*args)
    assert "sealed-position-leakage" in result["decision"]["reasons"]
    assert "missing-serving-evidence" in result["decision"]["reasons"]
    assert result["decision"]["status"] == "not-qualified"
    jsonschema.validate(result, core.read_json(core.ROOT / "docs/benchmark/report-schema-v1.json"))


@pytest.mark.parametrize(
    "field",
    [
        "candidate_manifest_sha256",
        "final_dataset_sha256",
        "development_dataset_sha256",
        "benchmark_sha256",
    ],
)
def test_seal_binding(tmp_path, field):
    args = final_inputs(tmp_path)
    seal = core.read_json(args[3])
    seal[field] = "e" * 64
    write_json(args[3], seal)
    with pytest.raises(ValueError, match="mismatch"):
        run_final(*args)


def test_seal_byte_digest_and_synthetic_final_rejected(tmp_path):
    dev, final, model, seal = final_inputs(tmp_path)
    with pytest.raises(ValueError, match="seal digest mismatch"):
        core.evaluate(
            final, model, mode="final", development_dir=dev, seal_path=seal, seal_sha256="f" * 64
        )
    manifest, rows = core.load_dataset(final)
    manifest["kind"] = "synthetic"
    save_data(final, manifest, rows)
    record = core.read_json(seal)
    record["final_dataset_sha256"] = core.digest(manifest)
    write_json(seal, record)
    with pytest.raises(ValueError, match="owner-controlled"):
        run_final(dev, final, model, seal)


def test_sealed_game_alias_leakage_even_with_different_group(tmp_path):
    dev, final, model, seal = final_inputs(tmp_path)
    manifest, rows = core.load_dataset(final)
    rows[0]["game_id"] = "synthetic-game-0"
    rows[0]["result"] = 0.5  # Even excluded rows must be audited.
    save_data(final, manifest, rows)
    record = core.read_json(seal)
    record["final_dataset_sha256"] = core.digest(manifest)
    write_json(seal, record)
    with pytest.raises(ValueError, match="group leakage"):
        run_final(dev, final, model, seal)


def test_missing_accepted_reference_fails(tmp_path):
    args = final_inputs(tmp_path)
    seal = core.read_json(args[3])
    seal["accepted_references"] = ["a" * 64]
    write_json(args[3], seal)
    with pytest.raises(ValueError, match="reference inventory"):
        run_final(*args)


def test_serving_checks_and_measurements_fail_closed():
    evidence = {
        "schema": "playground-serving-evidence-v1",
        "probe_suite_sha256": kcp13.sha256_of(core.ROOT / "docs/benchmark/serving-probes-v1.json"),
        "candidate_manifest_sha256": "a" * 64,
        "raw_evidence_sha256": "b" * 64,
        "checks": {
            name: True
            for name in (
                "jvm_golden_parity",
                "torch_onnx_parity",
                "immediate_king_capture",
                "forced_loss",
                "piece_safety",
                "probability_bounds",
                "concurrency",
            )
        },
        "measurements": {"latency_p95_ms": 2, "rss_mb": 30},
    }
    seal = {"serving_limits": {"latency_p95_ms": 3, "rss_mb": 40}}
    assert core.serving_check(evidence, "a" * 64, seal) == []
    evidence["measurements"]["latency_p95_ms"] = 4
    evidence["checks"]["forced_loss"] = False
    assert set(core.serving_check(evidence, "a" * 64, seal)) == {
        "serving:latency_p95_ms",
        "serving:forced_loss",
    }
    evidence["measurements"]["rss_mb"] = float("nan")
    with pytest.raises(ValueError, match="measurement"):
        core.serving_check(evidence, "a" * 64, seal)


def test_cli_sanitized_error_and_exit_code(capsys, tmp_path):
    assert main(["--data", str(tmp_path / "PRIVATE-CREDENTIAL-PATH")]) == 2
    output = capsys.readouterr().out
    assert "PRIVATE" not in output
    assert "Traceback" not in output
    assert json.loads(output)["schema"] == "playground-error-v1"
    assert main(["--data", str(FIXTURE), "--mode", "final"]) == 2


def test_implementation_tamper_fails_seal(tmp_path, monkeypatch):
    args = final_inputs(tmp_path)
    monkeypatch.setattr(core, "implementation_digest", lambda: "f" * 64)
    with pytest.raises(ValueError, match="implementation mismatch"):
        run_final(*args)


def test_complete_final_eligibility_path_on_artificial_fixtures(tmp_path):
    dev = copy_data(tmp_path, "dev", kind="public-licensed")
    final = copy_data(tmp_path, "final", kind="owner-controlled", rename=True)
    data, rows = core.load_dataset(dev)
    bare = kcp13.load_golden().by_id()["kings-only-w"]
    for row in rows:
        row.update(fen=bare.fen, side=bare.side, features=bare.features.tolist())
    save_data(dev, data, rows)
    manifest, _ = core.load_dataset(final)
    # Disjoint authored positions; artificial outcomes make the gate mathematically reachable.
    probes = [
        p
        for p in kcp13.load_golden().probes
        if not p.id.startswith("sample-") and p.id != "kings-only-w"
    ]
    rows = []
    for i in range(40):
        for j, p in enumerate(probes):
            rows.append(
                {
                    "id": f"test-{i}-{j}",
                    "group_id": f"sealed-{i}-{j}",
                    "game_id": f"sealed-{i}-{j}",
                    "fen": p.fen,
                    "side": p.side,
                    "ply": 5 if i % 2 else 30,
                    "features": p.features.tolist(),
                    "result": int(p.features[5] > 0) if p.features[5] else i % 2,
                }
            )
    save_data(final, manifest, rows)
    model = candidate(tmp_path, dev, weight=2)
    candidate_digest = core.digest(core.read_json(model / "manifest.json"))
    seal = tmp_path / "seal.json"
    write_json(
        seal,
        {
            "schema": "playground-seal-v1",
            "implementation_sha256": core.implementation_digest(),
            "benchmark_sha256": core.digest(core.load_protocol()),
            "candidate_manifest_sha256": candidate_digest,
            "development_dataset_sha256": core.digest(data),
            "final_dataset_sha256": core.digest(manifest),
            "accepted_references": [],
            "promotion_reference": "no-information",
            "serving_limits": {"latency_p95_ms": 10, "rss_mb": 100},
        },
    )
    evidence = tmp_path / "evidence.json"
    write_json(
        evidence,
        {
            "schema": "playground-serving-evidence-v1",
            "probe_suite_sha256": kcp13.sha256_of(
                core.ROOT / "docs/benchmark/serving-probes-v1.json"
            ),
            "candidate_manifest_sha256": candidate_digest,
            "raw_evidence_sha256": "a" * 64,
            "checks": {
                name: True
                for name in (
                    "jvm_golden_parity",
                    "torch_onnx_parity",
                    "immediate_king_capture",
                    "forced_loss",
                    "piece_safety",
                    "probability_bounds",
                    "concurrency",
                )
            },
            "measurements": {"latency_p95_ms": 1, "rss_mb": 1},
        },
    )
    report = run_final(dev, final, model, seal, evidence_path=evidence)
    assert report["decision"] == {"status": "eligible-for-owner-review", "reasons": []}
    jsonschema.validate(report, core.read_json(core.ROOT / "docs/benchmark/report-schema-v1.json"))


def test_reference_inventory_is_evaluated_and_cannot_silently_fall_back(tmp_path):
    dev, final, model, seal = final_inputs(tmp_path)
    reference = candidate(tmp_path, dev, "reference", weight=0.05)
    ref_digest = core.digest(core.read_json(reference / "manifest.json"))
    record = core.read_json(seal)
    record.update(accepted_references=[ref_digest], promotion_reference=ref_digest)
    write_json(seal, record)
    report = run_final(dev, final, model, seal, reference_dirs=[reference])
    assert report["baseline"] == ref_digest
    assert set(report["baselines"]) == {ref_digest, "no-information"}
    record["promotion_reference"] = "no-information"
    write_json(seal, record)
    with pytest.raises(ValueError, match="deployable reference required"):
        run_final(dev, final, model, seal, reference_dirs=[reference])


@pytest.mark.parametrize("repeats", [0, -1, 10001, 1.5])
def test_bootstrap_repeat_budget(repeats):
    with pytest.raises(ValueError, match="bootstrap repeats"):
        confidence([0, 1], [0.1, 0.9], ["a", "b"], repeats=repeats)


def test_cli_output_cannot_overwrite_existing_file(tmp_path, capsys):
    output = tmp_path / "keep.json"
    output.write_text("preserve this")
    assert main(["--data", str(FIXTURE), "--output", str(output)]) == 2
    assert output.read_text() == "preserve this"
    assert "invalid-benchmark-input" in capsys.readouterr().out
