"""Assemble the serving evidence the sealed benchmark refuses to run without.

The document is half measurement and half attestation, and the split is not negotiable.
`docs/benchmark/serving-probes-v1.json` states that the booleans attest reviewed external evidence
and that the CLI does not run the JVM or attest its truth: the terminal-capture probe, the
forced-loss probe, the concurrency replay and the two service measurements only exist against a
running evaluator, and nothing here may invent them.

Everything else is arithmetic, and arithmetic asserted by hand is how a document ends up attesting
what nobody measured. So this computes what it can — Torch/ONNX parity over every golden vector and
every qualification row, raw probability bounds before any serving clamp, the golden half of the
piece-safety probe, and the JVM parity error once the owner supplies the service's own responses —
carries what it cannot, and refuses to write a document for an observation nobody made.

The manifest-bound training model is recovered rather than stored: the serving contract fixes the
package at three files, and the packaging path is byte-for-byte reproducible from its own
provenance, so the model is rebuilt through the packager's own code and rejected unless its export
is identical to the shipped bytes.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from dicechess_training.ablation.export import export_candidate
from dicechess_training.ablation.runner import predict as torch_predict
from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.candidate.build import (
    MODEL_FILE,
    CandidateConfig,
    train_candidate,
)
from dicechess_training.contracts import kcp13
from dicechess_training.publish import stage, write_once

EVIDENCE_SCHEMA = "playground-serving-evidence-v1"
OBSERVATIONS_SCHEMA = "playground-serving-observations-v1"
PROBE_SUITE = "docs/benchmark/serving-probes-v1.json"

#: From the probe suite's own `checks`. Restated as constants so a tolerance can never be read
#: from a document the tool is in the middle of producing.
TORCH_ONNX_TOLERANCE = 1e-6
JVM_PARITY_TOLERANCE = 1e-5

#: Booleans only a running evaluator can establish. Never defaulted, inferred or computed.
ATTESTED_CHECKS = ("immediate_king_capture", "forced_loss", "concurrency")

#: Probes the suite names for the piece-safety check.
PIECE_SAFETY_PROBES = ("rook-queen-attack-w", "queen-en-prise-w")

ROOT = benchmark_core.ROOT


class ServingEvidenceError(ValueError):
    """Evidence could not be assembled. Messages never carry paths, hosts or measurements."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ServingEvidenceError(message)


def _is_number(value: Any) -> bool:
    """A real number, not a boolean wearing one: `isinstance(True, int)` is true in Python."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def authored_probes() -> list:
    """Golden probes the suite considers authored: the archive-derived samples are excluded."""
    return [p for p in kcp13.load_golden().probes if not p.id.startswith("sample-")]


def raw_outputs(model_path: str | Path, features: np.ndarray) -> np.ndarray:
    """Model output as the graph emits it, with no clamp and no reshaping.

    Deliberately not `kcp13.predict`, which clamps to [0,1] the way the evaluator does. The bounds
    check asks whether the raw output is already inside that interval, and a clamped reading can
    only ever answer yes.
    """
    import onnxruntime as ort

    x = np.asarray(features, dtype=np.float32)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (output,) = session.run([kcp13.OUTPUT_NAME], {kcp13.INPUT_NAME: x})
    return np.asarray(output)


def config_from_provenance(provenance: dict[str, str], model_id: str) -> CandidateConfig:
    """Rebuild the packager's configuration from the record the package carries.

    Every field is read back, none is assumed: a configuration reconstructed from defaults can
    only ever recover a candidate that was built with defaults, and would reject every other one
    as a digest mismatch it cannot explain. A package that predates the record is refused rather
    than guessed at.
    """
    _require("config" in provenance, "the package carries no configuration record to recover from")
    try:
        record = json.loads(provenance["config"])
    except ValueError as error:
        raise ServingEvidenceError("the configuration record is not readable") from error
    _require(isinstance(record, dict), "the configuration record is not a mapping")
    # Sequence fields travel through JSON as lists and the configuration declares them as tuples;
    # `as_record` turns them back into lists, so the digest is unaffected either way.
    record = {
        key: tuple(value) if key in ("hidden_dims", "epoch_candidates") else value
        for key, value in record.items()
    }
    # The record omits the model identity on purpose; the manifest is where it lives.
    record["model_id"] = model_id
    try:
        config = CandidateConfig(**record)
    except TypeError as error:
        raise ServingEvidenceError(
            "the configuration record does not describe a candidate"
        ) from error
    # The record now says what `config_sha256` only proved, so the two must agree before a single
    # weight is fitted — otherwise a tampered record would be trained from and only be caught
    # afterwards, by a digest comparison whose failure no longer explains itself.
    _require(
        benchmark_core.digest(config.as_record()) == provenance["config_sha256"],
        "the configuration record does not match the digest the package declares",
    )
    # Provenance states the seed and the calibration rule in their own right as well. A package
    # that disagrees with itself is not one to recover a model from.
    _require(
        str(config.seed) == str(provenance["seed"])
        and config.probability_calibration == provenance.get("probability_calibration", "none"),
        "the package disagrees with its own configuration record",
    )
    return config


def recover_model(candidate_dir: str | Path, training_data_dir: str | Path):
    """Recover the manifest-bound training model by reproducing the package.

    Returns the model only when the reproduction is byte-identical to the shipped artifact, so a
    parity check can never be run against a model that is merely similar to the one deployed.

    A package built before #59 cannot be recovered here. Until that fix, the exporter wrote the
    absolute source path and line number of the traced `forward` into every node, so its bytes
    depended on the checkout's location on disk — a rebuild reproduces the model and not the file.
    Such a package has to be re-issued rather than argued with.
    """
    candidate_dir = Path(candidate_dir)
    data_manifest, _ = benchmark_core.load_dataset(training_data_dir)
    try:
        # Admitted exactly as the benchmark admits it, before anything is computed with it: that
        # verifies the shipped model against its own manifest and the ONNX tensor contract, so a
        # swapped artifact is refused here rather than caught later by a check it happens to fail.
        manifest = benchmark_core.load_candidate(
            candidate_dir, data_manifest, benchmark_core.load_protocol()
        )
    except ValueError as error:
        raise ServingEvidenceError("the package is not one the benchmark would admit") from error
    provenance = manifest["provenance"]
    config = config_from_provenance(provenance, manifest["modelId"])

    trained = train_candidate(training_data_dir, config)
    for key in ("config_sha256", "code_sha256", "training_data_sha256", "training_groups_sha256"):
        _require(
            trained.provenance[key] == provenance[key],
            f"recovered candidate disagrees with the package on {key}",
        )

    with tempfile.TemporaryDirectory() as staging:
        replay = Path(staging) / MODEL_FILE
        export_candidate(trained.model, kcp13.SCHEMA_ID, replay)
        _require(
            kcp13.sha256_of(replay) == manifest["modelSha256"],
            "recovered model does not reproduce the packaged bytes",
        )
    return trained.model, manifest


def _qualification_features(dataset_dirs) -> tuple[np.ndarray, list[dict]]:
    """Every row of every supplied dataset, in the order its own bundle declares."""
    features, accounting = [], []
    for directory in dataset_dirs:
        manifest, rows = benchmark_core.load_dataset(directory)
        features.append(np.asarray([r["features"] for r in rows], dtype=np.float32))
        accounting.append({"dataset_sha256": benchmark_core.digest(manifest), "rows": len(rows)})
    _require(bool(features), "at least one qualification dataset is required")
    return np.concatenate(features), accounting


def _torch_onnx_parity(model, model_path: Path, golden: np.ndarray, rows: np.ndarray) -> dict:
    observations = {}
    worst = 0.0
    for name, matrix in (("golden", golden), ("qualification_rows", rows)):
        torch_p = torch_predict(model, matrix).astype(np.float64)
        onnx_p = kcp13.predict(model_path, matrix).astype(np.float64)
        _require(len(torch_p) == len(matrix) and len(onnx_p) == len(matrix), "parity lost rows")
        diff = float(np.max(np.abs(torch_p - onnx_p)))
        observations[name] = {"rows": int(len(matrix)), "max_abs_diff": diff}
        worst = max(worst, diff)
    observations["max_abs_diff"] = worst
    observations["tolerance"] = TORCH_ONNX_TOLERANCE
    return {"passed": worst <= TORCH_ONNX_TOLERANCE, "observations": observations}


def _probability_bounds(model_path: Path, matrices: dict[str, np.ndarray]) -> dict:
    observations, passed = {}, True
    for name, matrix in matrices.items():
        output = raw_outputs(model_path, matrix)
        shape_ok = output.shape == (len(matrix), 1)
        flat = output.reshape(-1).astype(np.float64) if output.size else output.reshape(-1)
        finite = bool(np.isfinite(flat).all()) if flat.size else True
        # Range is only meaningful once the values are finite; a NaN compares false either way.
        in_range = bool(finite and flat.size and ((flat >= 0.0) & (flat <= 1.0)).all())
        observations[name] = {
            "rows": int(len(matrix)),
            "shape_matches_contract": shape_ok,
            "all_finite": finite,
            "within_unit_interval": in_range,
            "min": float(flat.min()) if finite and flat.size else None,
            "max": float(flat.max()) if finite and flat.size else None,
        }
        passed = passed and shape_ok and finite and in_range
    return {"passed": passed, "observations": observations}


def _jvm_golden_parity(model_path: Path, responses: dict[str, float]) -> dict:
    probes = authored_probes()
    missing = sorted({p.id for p in probes} - set(responses))
    extra = sorted(set(responses) - {p.id for p in probes})
    _require(not missing, "the service did not answer every authored golden probe")
    _require(not extra, "the service answered a probe the authored corpus does not contain")
    ours = kcp13.predict(model_path, np.stack([p.features for p in probes])).astype(np.float64)
    errors = {
        probe.id: abs(float(responses[probe.id]) - float(value))
        for probe, value in zip(probes, ours, strict=True)
    }
    worst = max(errors.values())
    return {
        "passed": worst <= JVM_PARITY_TOLERANCE,
        "observations": {
            "probes": len(probes),
            "max_abs_error": worst,
            "tolerance": JVM_PARITY_TOLERANCE,
            "per_probe_abs_error": errors,
        },
    }


def _piece_safety(model_path: Path, matched_alternatives_reviewed: bool) -> dict:
    """The half the corpus can settle; the matched alternatives stay owner-reviewed.

    The suite is explicit that probability ordering across unmatched boards is not an acceptance
    criterion, so nothing here compares one position against another.
    """
    by_id = kcp13.load_golden().by_id()
    queen_columns = [i for i, name in enumerate(kcp13.COLUMN_NAMES) if name.startswith("queen_")]
    observations, computed = {}, True
    for base in PIECE_SAFETY_PROBES:
        twin = f"{base}-twin"
        _require(
            base in by_id and twin in by_id, "the golden corpus is missing a piece-safety twin"
        )
        features = np.stack([by_id[base].features, by_id[twin].features])
        probabilities = kcp13.predict(model_path, features).astype(np.float64)
        twins_share_features = bool(np.array_equal(features[0], features[1]))
        queen_signal = bool(np.any(features[0][queen_columns] > 0.0))
        difference = float(abs(probabilities[0] - probabilities[1]))
        observations[base] = {
            "twins_share_feature_vector": twins_share_features,
            "queen_columns_carry_signal": queen_signal,
            "twin_probability_abs_diff": difference,
        }
        computed = computed and twins_share_features and queen_signal
        computed = computed and difference <= JVM_PARITY_TOLERANCE
    observations["matched_alternatives_reviewed"] = matched_alternatives_reviewed
    return {
        "passed": bool(computed and matched_alternatives_reviewed),
        "observations": observations,
    }


def load_observations(path: str | Path) -> dict[str, Any]:
    """Read the owner's service observations.

    Refuses anything the document cannot honestly be built from: a missing attestation is not a
    failed check, it is an absent one, and the difference must not be flattened into a `false`.
    """
    observations = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(observations.get("schema") == OBSERVATIONS_SCHEMA, "unsupported observations schema")
    try:
        # The benchmark owns what a digest looks like; only the error type is this module's.
        benchmark_core.require_sha(observations.get("concurrency_workload_sha256"))
    except ValueError as error:
        raise ServingEvidenceError("the concurrency workload digest is unusable") from error
    attested = observations.get("attested", {})
    for check in (*ATTESTED_CHECKS, "piece_safety_matched_alternatives"):
        _require(
            isinstance(attested.get(check), bool),
            "an observation only a reviewed evaluator can supply is missing",
        )
    responses = observations.get("jvm_golden_probabilities", {})
    _require(isinstance(responses, dict) and bool(responses), "service probe responses are missing")
    for value in responses.values():
        # `bool` is a subclass of `int`, so a JSON `true` passes a numeric check and would be
        # written out as 1.0. A probability nobody measured must not become one that reads as
        # measured, so booleans are refused rather than coerced.
        _require(
            _is_number(value) and np.isfinite(value),
            "a service probe response is not a finite number",
        )
    for measurement in ("latency_p95_ms", "rss_mb"):
        value = observations.get("measurements", {}).get(measurement)
        _require(
            _is_number(value) and np.isfinite(value) and value >= 0,
            "a service measurement is missing or unusable",
        )
    return observations


def build_evidence(
    candidate_dir: str | Path,
    training_data_dir: str | Path,
    qualification_data_dirs,
    observations_path: str | Path,
    evidence_path: str | Path,
    raw_evidence_path: str | Path,
) -> dict[str, Any]:
    """Write the raw observations and the evidence document, or raise and write neither."""
    observations = load_observations(observations_path)
    candidate_dir = Path(candidate_dir)
    model_path = candidate_dir / MODEL_FILE

    model, manifest = recover_model(candidate_dir, training_data_dir)
    golden = np.stack([p.features for p in authored_probes()])
    rows, accounting = _qualification_features(qualification_data_dirs)

    results = {
        "torch_onnx_parity": _torch_onnx_parity(model, model_path, golden, rows),
        "probability_bounds": _probability_bounds(
            model_path, {"golden": golden, "qualification_rows": rows}
        ),
        "jvm_golden_parity": _jvm_golden_parity(
            model_path, observations["jvm_golden_probabilities"]
        ),
        "piece_safety": _piece_safety(
            model_path, observations["attested"]["piece_safety_matched_alternatives"]
        ),
    }
    checks = {name: bool(result["passed"]) for name, result in results.items()}
    checks.update({name: bool(observations["attested"][name]) for name in ATTESTED_CHECKS})

    raw = {
        "schema": "playground-serving-raw-v1",
        "candidate_manifest_sha256": benchmark_core.digest(manifest),
        "model_recovered_from_provenance": True,
        "qualification_datasets": accounting,
        "computed": {name: result["observations"] for name, result in results.items()},
        "attested": observations["attested"],
        "measurements": observations["measurements"],
        "owner_raw": observations.get("raw"),
    }
    raw_evidence_path, evidence_path = Path(raw_evidence_path), Path(evidence_path)
    _require(
        raw_evidence_path.resolve() != evidence_path.resolve(),
        "the two documents cannot be written to the same location",
    )
    for destination in (raw_evidence_path, evidence_path):
        # A friendly early refusal; `_publish` is what actually makes the guarantee.
        _require(not destination.exists(), "an output file already exists")

    # The raw file is serialised first because the evidence has to carry its digest, and a digest
    # of something that was never written is the one thing this document may not contain.
    with contextlib.ExitStack() as stack:
        staged_raw = stage(
            raw_evidence_path,
            json.dumps(raw, indent=2, sort_keys=True, allow_nan=False) + "\n",
            stack,
        )
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "candidate_manifest_sha256": benchmark_core.digest(manifest),
            "probe_suite_sha256": kcp13.sha256_of(ROOT / PROBE_SUITE),
            "concurrency_workload_sha256": observations["concurrency_workload_sha256"],
            "raw_evidence_sha256": kcp13.sha256_of(staged_raw),
            "checks": checks,
            "measurements": {
                "latency_p95_ms": float(observations["measurements"]["latency_p95_ms"]),
                "rss_mb": float(observations["measurements"]["rss_mb"]),
            },
        }
        staged_evidence = stage(
            evidence_path,
            json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
            stack,
        )
        write_once([(staged_raw, raw_evidence_path), (staged_evidence, evidence_path)])
    return {
        "evidence": evidence,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }
