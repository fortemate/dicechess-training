"""Serving evidence: what the tool may compute, and what it may only carry.

The document decides whether a candidate qualifies, so the tests that matter most are the ones
proving the tool cannot assert something nobody observed, and cannot quietly turn an absent
attestation into a failed one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dicechess_training.benchmark import core
from dicechess_training.candidate import CandidateConfig, build_candidate
from dicechess_training.contracts import kcp13
from dicechess_training.serving import ServingEvidenceError, build_evidence, recover_model
from dicechess_training.serving.__main__ import main
from dicechess_training.serving.evidence import (
    EVIDENCE_SCHEMA,
    JVM_PARITY_TOLERANCE,
    OBSERVATIONS_SCHEMA,
    PROBE_SUITE,
    authored_probes,
    raw_outputs,
)

FIXTURE = core.ROOT / "tests/fixtures/benchmark"
WORKLOAD = "a" * 64


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    directory = tmp_path_factory.mktemp("serving") / "package"
    build_candidate(FIXTURE, directory, CandidateConfig(seed=11, model_id="serving-test-candidate"))
    return directory


def _service_responses(candidate_dir, offset=0.0):
    """What a faithful evaluator would answer for every authored probe."""
    probes = authored_probes()
    ours = kcp13.predict(candidate_dir / "model.onnx", np.stack([p.features for p in probes]))
    return {probe.id: float(value) + offset for probe, value in zip(probes, ours, strict=True)}


def _observations(candidate_dir, **overrides):
    document = {
        "schema": OBSERVATIONS_SCHEMA,
        "concurrency_workload_sha256": WORKLOAD,
        "jvm_golden_probabilities": _service_responses(candidate_dir),
        "attested": {
            "immediate_king_capture": True,
            "forced_loss": True,
            "concurrency": True,
            "piece_safety_matched_alternatives": True,
        },
        "measurements": {"latency_p95_ms": 12.5, "rss_mb": 420.0},
        "raw": {"admitted_requests": 1000, "errors": 0},
    }
    document.update(overrides)
    return document


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _staged(tmp_path, observations, name="run"):
    """Write the observations file. Setup, never the call under test."""
    return _write(tmp_path / f"{name}-observations.json", observations)


def _build(tmp_path, candidate_dir, observations_path, name="run"):
    """The single call a refusal test puts inside `pytest.raises`."""
    return build_evidence(
        candidate_dir,
        FIXTURE,
        [FIXTURE],
        observations_path,
        tmp_path / f"{name}-evidence.json",
        tmp_path / f"{name}-raw.json",
    )


def _run(tmp_path, candidate_dir, observations, name="run"):
    return _build(tmp_path, candidate_dir, _staged(tmp_path, observations, name), name)


def test_a_faithful_service_produces_a_document_every_check_passes(tmp_path, candidate):
    result = _run(tmp_path, candidate, _observations(candidate))
    evidence = result["evidence"]
    assert result["failed_checks"] == []
    assert evidence["schema"] == EVIDENCE_SCHEMA
    assert set(evidence["checks"]) == {
        "torch_onnx_parity",
        "probability_bounds",
        "jvm_golden_parity",
        "piece_safety",
        "immediate_king_capture",
        "forced_loss",
        "concurrency",
    }
    assert all(evidence["checks"].values())


def test_the_document_carries_the_digests_the_benchmark_verifies(tmp_path, candidate):
    result = _run(tmp_path, candidate, _observations(candidate))
    evidence = result["evidence"]
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    assert evidence["candidate_manifest_sha256"] == core.digest(manifest)
    assert evidence["probe_suite_sha256"] == kcp13.sha256_of(core.ROOT / PROBE_SUITE)
    assert evidence["concurrency_workload_sha256"] == WORKLOAD
    assert evidence["raw_evidence_sha256"] == kcp13.sha256_of(tmp_path / "run-raw.json")


def test_the_benchmarks_own_serving_check_accepts_it(tmp_path, candidate):
    """The only acceptance test that matters: the consumer agrees to read it."""
    result = _run(tmp_path, candidate, _observations(candidate))
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    seal = {
        "concurrency_workload_sha256": WORKLOAD,
        "serving_limits": {"latency_p95_ms": 50.0, "rss_mb": 1024.0},
    }
    assert core.serving_check(result["evidence"], core.digest(manifest), seal) == []


def test_measurements_beyond_the_sealed_limits_are_reported_by_the_consumer(tmp_path, candidate):
    result = _run(tmp_path, candidate, _observations(candidate))
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    seal = {
        "concurrency_workload_sha256": WORKLOAD,
        "serving_limits": {"latency_p95_ms": 1.0, "rss_mb": 1.0},
    }
    assert sorted(core.serving_check(result["evidence"], core.digest(manifest), seal)) == [
        "serving:latency_p95_ms",
        "serving:rss_mb",
    ]


@pytest.mark.parametrize(
    "check", ["immediate_king_capture", "forced_loss", "concurrency", "piece_safety"]
)
def test_a_failing_service_observation_becomes_a_failing_check(tmp_path, candidate, check):
    attested = _observations(candidate)["attested"]
    attested["piece_safety_matched_alternatives" if check == "piece_safety" else check] = False
    result = _run(tmp_path, candidate, _observations(candidate, attested=attested), name=check)
    assert result["evidence"]["checks"][check] is False
    assert check in result["failed_checks"]


@pytest.mark.parametrize(
    "missing",
    ["immediate_king_capture", "forced_loss", "concurrency", "piece_safety_matched_alternatives"],
)
def test_an_absent_attestation_refuses_the_document(tmp_path, candidate, missing):
    """Absent is not false. The tool may not decide a question nobody asked the evaluator."""
    attested = _observations(candidate)["attested"]
    del attested[missing]
    observations = _staged(tmp_path, _observations(candidate, attested=attested), missing)
    with pytest.raises(ServingEvidenceError, match="only a reviewed evaluator can supply"):
        _build(tmp_path, candidate, observations, missing)
    assert not (tmp_path / f"{missing}-evidence.json").exists()
    assert not (tmp_path / f"{missing}-raw.json").exists()


def test_a_missing_measurement_refuses_the_document(tmp_path, candidate):
    observations = _staged(tmp_path, _observations(candidate, measurements={"rss_mb": 1.0}))
    with pytest.raises(ServingEvidenceError, match="measurement is missing"):
        _build(tmp_path, candidate, observations)


def test_a_service_that_skipped_a_probe_refuses_the_document(tmp_path, candidate):
    responses = _service_responses(candidate)
    responses.pop(next(iter(responses)))
    observations = _staged(tmp_path, _observations(candidate, jvm_golden_probabilities=responses))
    with pytest.raises(ServingEvidenceError, match="did not answer every authored golden probe"):
        _build(tmp_path, candidate, observations)


def test_a_service_answering_an_unknown_probe_refuses_the_document(tmp_path, candidate):
    responses = _service_responses(candidate)
    responses["invented-probe"] = 0.5
    observations = _staged(tmp_path, _observations(candidate, jvm_golden_probabilities=responses))
    with pytest.raises(ServingEvidenceError, match="corpus does not contain"):
        _build(tmp_path, candidate, observations)


def test_a_service_outside_the_parity_tolerance_fails_that_check(tmp_path, candidate):
    drifted = _service_responses(candidate, offset=10 * JVM_PARITY_TOLERANCE)
    result = _run(tmp_path, candidate, _observations(candidate, jvm_golden_probabilities=drifted))
    assert result["evidence"]["checks"]["jvm_golden_parity"] is False
    assert result["failed_checks"] == ["jvm_golden_parity"]


def test_an_unusable_workload_digest_refuses_the_document(tmp_path, candidate):
    observations = _staged(tmp_path, _observations(candidate, concurrency_workload_sha256="nope"))
    with pytest.raises(ServingEvidenceError):
        _build(tmp_path, candidate, observations)


def test_an_unknown_observations_schema_refuses_the_document(tmp_path, candidate):
    observations = _staged(tmp_path, _observations(candidate, schema="something-else"))
    with pytest.raises(ServingEvidenceError, match="unsupported observations schema"):
        _build(tmp_path, candidate, observations)


def _repackage(directory, candidate, mutate):
    """A copy of the package with its manifest mutated and its model left alone."""
    directory.mkdir()
    (directory / "model.onnx").write_bytes((candidate / "model.onnx").read_bytes())
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    mutate(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_the_model_is_recovered_only_when_it_reproduces_the_packaged_bytes(tmp_path, candidate):
    model, manifest = recover_model(candidate, FIXTURE)
    assert manifest["modelSha256"] == kcp13.sha256_of(candidate / "model.onnx")
    assert model is not None


def test_a_swapped_model_is_refused_before_anything_is_computed_with_it(tmp_path, candidate):
    """The manifest alone is not the artifact. A replaced model must not reach the checks."""
    swapped = tmp_path / "swapped"
    swapped.mkdir()
    (swapped / "manifest.json").write_bytes((candidate / "manifest.json").read_bytes())
    original = (candidate / "model.onnx").read_bytes()
    (swapped / "model.onnx").write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    with pytest.raises(ServingEvidenceError, match="not one the benchmark would admit"):
        recover_model(swapped, FIXTURE)


def test_a_package_without_a_configuration_record_is_refused_rather_than_guessed(
    tmp_path, candidate
):
    directory = _repackage(
        tmp_path / "no-config", candidate, lambda m: m["provenance"].pop("config")
    )
    with pytest.raises(ServingEvidenceError, match="no configuration record"):
        recover_model(directory, FIXTURE)


def test_a_package_that_disagrees_with_its_own_record_is_refused(tmp_path, candidate):
    directory = _repackage(
        tmp_path / "inconsistent", candidate, lambda m: m["provenance"].__setitem__("seed", "999")
    )
    with pytest.raises(ServingEvidenceError, match="disagrees with its own configuration record"):
        recover_model(directory, FIXTURE)


def test_a_tampered_configuration_record_is_refused_before_training(tmp_path, candidate):
    def rewrite(manifest):
        record = json.loads(manifest["provenance"]["config"])
        record["learning_rate"] = 0.5
        manifest["provenance"]["config"] = json.dumps(record, sort_keys=True, separators=(",", ":"))

    directory = _repackage(tmp_path / "tampered-config", candidate, rewrite)
    with pytest.raises(ServingEvidenceError, match="does not match the digest"):
        recover_model(directory, FIXTURE)


def test_a_candidate_built_with_non_default_settings_is_recoverable(tmp_path):
    """The reason the record exists: recovery must not be limited to default configurations."""
    directory = tmp_path / "non-default"
    config = CandidateConfig(
        seed=5,
        model_id="non-default-candidate",
        hidden_dims=(8, 8),
        learning_rate=5e-3,
        batch_size=64,
        epoch_candidates=(2, 3),
        inner_cutoff=6500,
    )
    build_candidate(FIXTURE, directory, config)
    model, manifest = recover_model(directory, FIXTURE)
    assert model is not None
    record = json.loads(manifest["provenance"]["config"])
    assert record["hidden_dims"] == [8, 8]
    assert record["inner_cutoff"] == 6500
    # The identity is the manifest's, and is not duplicated into a field that travels beside it.
    assert "model_id" not in record
    assert manifest["modelId"] == "non-default-candidate"


def test_bounds_are_read_before_the_serving_clamp(candidate):
    """`kcp13.predict` clamps, so a clamped reading could only ever answer yes."""
    matrix = np.stack([p.features for p in authored_probes()])
    output = raw_outputs(candidate / "model.onnx", matrix)
    assert output.shape == (len(matrix), 1)
    assert np.isfinite(output).all()


def test_cli_reports_digests_and_failed_checks_but_no_measurements(tmp_path, candidate, capsys):
    observations = _write(tmp_path / "cli-observations.json", _observations(candidate))
    code = main(
        [
            "--candidate",
            str(candidate),
            "--training-data",
            str(FIXTURE),
            "--qualification-data",
            str(FIXTURE),
            "--observations",
            str(observations),
            "--output",
            str(tmp_path / "cli-evidence.json"),
            "--raw-output",
            str(tmp_path / "cli-raw.json"),
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["failed_checks"] == []
    assert printed["checks_passed"] == printed["checks_total"] == 7
    assert "measurements" not in printed
    assert "latency_p95_ms" not in json.dumps(printed)
    # The measurements still reach the document the owner retains.
    assert (
        json.loads((tmp_path / "cli-evidence.json").read_text())["measurements"]["rss_mb"] == 420.0
    )


def test_cli_exits_nonzero_when_a_check_failed(tmp_path, candidate, capsys):
    attested = _observations(candidate)["attested"]
    attested["concurrency"] = False
    observations = _write(
        tmp_path / "fail-observations.json", _observations(candidate, attested=attested)
    )
    code = main(
        [
            "--candidate",
            str(candidate),
            "--training-data",
            str(FIXTURE),
            "--qualification-data",
            str(FIXTURE),
            "--observations",
            str(observations),
            "--output",
            str(tmp_path / "fail-evidence.json"),
            "--raw-output",
            str(tmp_path / "fail-raw.json"),
        ]
    )
    assert code == 1
    assert json.loads(capsys.readouterr().out)["failed_checks"] == ["concurrency"]


def test_cli_refuses_invalid_arguments():
    assert main([]) == 2


def test_a_boolean_is_not_accepted_where_a_number_is_required(tmp_path, candidate):
    """`isinstance(True, int)` is true in Python, so `true` would have been written out as 1.0."""
    responses = _service_responses(candidate)
    responses[next(iter(responses))] = True
    observations = _staged(tmp_path, _observations(candidate, jvm_golden_probabilities=responses))
    with pytest.raises(ServingEvidenceError, match="not a finite number"):
        _build(tmp_path, candidate, observations)

    measured = _staged(
        tmp_path,
        _observations(candidate, measurements={"latency_p95_ms": True, "rss_mb": 420.0}),
        "bool-measurement",
    )
    with pytest.raises(ServingEvidenceError, match="measurement is missing or unusable"):
        _build(tmp_path, candidate, measured, "bool-measurement")


def test_a_non_finite_measurement_is_refused(tmp_path, candidate):
    observations = _observations(candidate)
    observations["measurements"] = {"latency_p95_ms": float("inf"), "rss_mb": 420.0}
    # `json.dumps` writes Infinity by default, which `json.loads` reads back as a float.
    staged = _write(tmp_path / "inf-observations.json", observations)
    with pytest.raises(ServingEvidenceError, match="measurement is missing or unusable"):
        _build(tmp_path, candidate, staged, "inf")


def test_an_existing_output_is_never_overwritten(tmp_path, candidate):
    for occupied in ("run-raw.json", "run-evidence.json"):
        directory = tmp_path / occupied.split(".")[0]
        directory.mkdir()
        (directory / occupied).write_text("previous run", encoding="utf-8")
        observations = _staged(directory, _observations(candidate))
        with pytest.raises(ServingEvidenceError, match="already exists"):
            _build(directory, candidate, observations)
        assert (directory / occupied).read_text(encoding="utf-8") == "previous run"


def test_a_refused_run_leaves_no_staging_directory_behind(tmp_path, candidate):
    attested = _observations(candidate)["attested"]
    del attested["concurrency"]
    observations = _staged(tmp_path, _observations(candidate, attested=attested), "aborted")
    with pytest.raises(ServingEvidenceError):
        _build(tmp_path, candidate, observations, "aborted")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["aborted-observations.json"]
