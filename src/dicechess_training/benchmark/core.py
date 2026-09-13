"""Digest-bound inputs, benchmark execution, and fail-closed qualification gates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from dicechess_training.contracts import kcp13

from .metrics import confidence, scores
from .splits import assignments, leakage, position_key

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/benchmark/protocol-v1.json"
SHA = re.compile(r"[0-9a-f]{64}")
SLICES = (
    "phase:opening",
    "phase:middlegame",
    "phase:endgame",
    "side:w",
    "side:b",
    "material:behind",
    "material:balanced",
    "material:ahead",
    "target:decisive",
    "target:uncertain",
    "tactical:king_attack",
    "tactical:king_danger",
    "tactical:queen_attack",
    "tactical:queen_danger",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def implementation_digest():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [
        Path(kcp13.__file__),
        ROOT / "uv.lock",
        ROOT / "docs/benchmark/serving-probes-v1.json",
    ]
    return digest({str(p.relative_to(ROOT)): kcp13.sha256_of(p) for p in paths})


def check_training_identity(candidate, manifest, rows):
    expected = train_identity(manifest, rows)
    require(
        all(candidate["provenance"][key] == value for key, value in expected.items()),
        "artifact training identity mismatch",
    )


def read_json(path):
    def duplicate_safe(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=duplicate_safe)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_sha(value):
    require(isinstance(value, str) and SHA.fullmatch(value) is not None, "invalid digest")


def load_protocol():
    return read_json(PROTOCOL)


def load_dataset(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    require(manifest["schema"] == "playground-rows-v1", "unsupported dataset schema")
    require(
        re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", manifest["version"]) is not None,
        "invalid dataset version",
    )
    require(manifest["perspective"] == "side-to-move", "wrong perspective")
    require(manifest["target"] == "decisive-game-outcome", "unsupported target")
    require(
        manifest["kind"] in ("synthetic", "public-licensed", "owner-controlled"),
        "unverified dataset origin",
    )
    for key in ("rows_sha256", "license_evidence_sha256", "source_sha256", "golden_sha256"):
        require_sha(manifest[key])
    require(manifest["license"] not in ("", "unknown", "unresolved"), "missing data terms")
    require(
        kcp13.sha256_of(directory / "license.txt") == manifest["license_evidence_sha256"],
        "data terms digest mismatch",
    )
    require(
        kcp13.sha256_of(directory / "rows.json") == manifest["rows_sha256"],
        "dataset digest mismatch",
    )
    require(manifest["feature_schema"] == kcp13.SCHEMA_ID, "unsupported features")
    require(manifest["columns"] == list(kcp13.COLUMN_NAMES), "wrong feature order")
    require(manifest["engine_version"] == kcp13.GOLDEN_ENGINE_VERSION, "unverified engine")
    require(
        manifest["golden_sha256"] == kcp13.sha256_of(kcp13.golden_path()), "golden digest mismatch"
    )
    rows = read_json(directory / "rows.json")
    require(isinstance(rows, list) and bool(rows), "empty dataset")
    ids = set()
    for row in rows:
        for key in ("id", "group_id", "game_id"):
            require(isinstance(row[key], str) and bool(row[key].strip()), "missing row identity")
        require(row["id"] not in ids, "duplicate row identity")
        ids.add(row["id"])
        fields = row["fen"].split()
        require(len(fields) in (4, 6) and fields[1] in ("w", "b"), "invalid canonical FEN")
        require(row["side"] == fields[1], "FEN perspective mismatch")
        require(isinstance(row["ply"], int) and row["ply"] >= 0, "invalid ply")
        require(row["result"] in (0, 0.5, 1), "invalid game result")
        placement = fields[0]
        require(placement.count("K") == placement.count("k") == 1, "terminal or invalid kings")
        ranks = placement.split("/")
        require(
            len(ranks) == 8
            and all(
                all(c in "12345678pnbrqkPNBRQK" for c in rank)
                and sum(int(c) if c.isdigit() else 1 for c in rank) == 8
                for rank in ranks
            ),
            "invalid board placement",
        )
        features = np.asarray(row["features"], dtype=np.float32)
        require(features.shape == (13,) and np.isfinite(features).all(), "invalid feature vector")
        require(
            np.allclose(
                features[:7], kcp13.material_block(row["fen"], row["side"]), rtol=0, atol=1e-6
            ),
            "material/perspective mismatch",
        )
    assignments(rows)  # Validate game/root ownership even for a sealed dataset.
    return manifest, rows


def load_candidate(directory, data_manifest, protocol):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    kcp13.validate_manifest(manifest, data_manifest["engine_version"])
    kcp13.verify_model_digest(directory / "model.onnx", manifest)
    kcp13.validate_onnx_contract(directory / "model.onnx")
    provenance = manifest["provenance"]
    for key in (
        "training_data_sha256",
        "training_groups_sha256",
        "config_sha256",
        "benchmark_sha256",
        "code_sha256",
    ):
        require_sha(provenance[key])
    require(provenance["benchmark_sha256"] == digest(protocol), "candidate protocol mismatch")
    require(provenance["engine_version"] == data_manifest["engine_version"], "extractor mismatch")
    require(isinstance(provenance["seed"], int), "missing training seed")
    require(provenance["perspective"] == "side-to-move", "candidate perspective mismatch")
    # Calibration must live in the graph. A manifest-only adjustment would diverge from predict.
    require(
        manifest.get("calibration", {}) in ({}, {"temperature": 1.0}),
        "calibration must be embedded in the graph",
    )
    return manifest


def train_identity(manifest, rows):
    groups = sorted({row["group_id"] for row in rows if assignments([row])[0] == "train"})
    return {"training_data_sha256": digest(manifest), "training_groups_sha256": digest(groups)}


def slice_names(row):
    f = row["features"]
    phase = "opening" if row["ply"] <= 10 else "endgame" if f[6] <= 20 else "middlegame"
    material = "behind" if f[5] < -1 else "ahead" if f[5] > 1 else "balanced"
    names = {f"phase:{phase}", f"side:{row['side']}", f"material:{material}", "target:decisive"}
    for i, name in enumerate(("king_attack", "king_danger", "queen_attack", "queen_danger"), 9):
        if f[i] > 0:
            names.add("tactical:" + name)
    return names


def probe_report(model_path, protocol):
    corpus = kcp13.load_golden()
    # Upstream fixture contains archive-derived examples. Use only authored synthetic probes.
    probes = [p for p in corpus.probes if not p.id.startswith("sample-")]
    values = kcp13.predict(model_path, np.stack([p.features for p in probes]))
    p = dict(zip((p.id for p in probes), values, strict=True))
    tol = protocol["probes"]["equal_tolerance"]
    checks = {}
    for name in p:
        if name.endswith("-twin"):
            checks[name] = abs(p[name] - p[name.removesuffix("-twin")]) <= tol
    checks["opening-canonical"] = abs(p["start-w"] - p["start-w-6field"]) <= tol
    checks["opening-side"] = abs(p["start-w"] - p["start-b"]) <= tol
    checks["ep-canonical"] = abs(p["ep-e6-w"] - p["ep-none-w"]) <= tol
    checks["opening-balanced"] = (
        abs(p["start-w"] - 0.5) <= protocol["probes"]["opening_distance_from_half"]
    )
    checks["material-order"] = p["knight-up-w"] > p["start-w"] > p["knight-down-b"]
    # Dice-free threats have roll-dependent capture chances, not certain win/loss labels.
    return {
        "checks": {key: bool(value) for key, value in checks.items()},
        "diagnostic_count": len(probes),
        "king_capture": "requires-serving-evidence",
        "piece_safety": "feature-invariants-and-twins; no-proven-outcome-order",
    }


def summarize(rows, p, reference, protocol):
    y = [r["result"] for r in rows]
    groups = [r["group_id"] for r in rows]
    return {
        "candidate": scores(y, p),
        "reference": scores(y, reference),
        "confidence": confidence(
            y,
            p,
            groups,
            reference,
            repeats=protocol["uncertainty"]["repeats"],
            seed=protocol["uncertainty"]["seed"],
        ),
    }


def gate_check(summary, protocol, *, critical_slice=False):
    rule = protocol["gate"]
    c, r, ci = summary["candidate"], summary["reference"], summary["confidence"]
    reasons = []
    minimum = rule["min_slice_groups"] if critical_slice else rule["min_groups"]
    if ci["groups"] < minimum:
        reasons.append("insufficient-groups")
    if critical_slice:
        if c["log_loss"] - r["log_loss"] > rule["max_slice_log_loss_regression"]:
            reasons.append("slice-log-loss-regression")
        if c["brier"] - r["brier"] > rule["max_slice_brier_regression"]:
            reasons.append("slice-brier-regression")
    else:
        if c["log_loss"] > r["log_loss"] * (1 - rule["min_relative_log_loss_gain"]):
            reasons.append("insufficient-relative-gain")
        if c["brier"] - r["brier"] > rule["max_brier_regression"]:
            reasons.append("brier-regression")
        if c["ece"] - r["ece"] > rule["max_ece_regression"]:
            reasons.append("calibration-regression")
        if ci["intervals"] is None or ci["intervals"]["log_loss_delta"][1] >= 0:
            reasons.append("uncertain-improvement")
    return reasons


def serving_check(evidence, candidate_digest, seal):
    """Verify digest-bound owner evidence; never infer service checks from Python predictions."""
    require(evidence["candidate_manifest_sha256"] == candidate_digest, "wrong serving candidate")
    require(evidence["schema"] == "playground-serving-evidence-v1", "unknown serving evidence")
    require(
        evidence["probe_suite_sha256"]
        == kcp13.sha256_of(ROOT / "docs/benchmark/serving-probes-v1.json"),
        "serving probe suite mismatch",
    )
    reasons = []
    for check in (
        "jvm_golden_parity",
        "torch_onnx_parity",
        "immediate_king_capture",
        "forced_loss",
        "piece_safety",
        "probability_bounds",
        "concurrency",
    ):
        if evidence["checks"].get(check) is not True:
            reasons.append("serving:" + check)
    for measurement in ("latency_p95_ms", "rss_mb"):
        value = evidence["measurements"][measurement]
        limit = seal["serving_limits"][measurement]
        require(
            isinstance(value, (int, float)) and np.isfinite(value) and value >= 0,
            "invalid serving measurement",
        )
        require(
            isinstance(limit, (int, float)) and np.isfinite(limit) and limit > 0,
            "invalid serving limit",
        )
        if value > limit:
            reasons.append("serving:" + measurement)
    require_sha(evidence["raw_evidence_sha256"])
    return reasons


def evaluate(
    data_dir,
    candidate_dir=None,
    *,
    mode="development",
    development_dir=None,
    seal_path=None,
    seal_sha256=None,
    reference_dirs=(),
    evidence_path=None,
):
    protocol = load_protocol()
    data, all_rows = load_dataset(data_dir)
    rows = [r for r in all_rows if r["result"] != 0.5]
    require(bool(rows), "no decisive outcomes")
    candidate = load_candidate(candidate_dir, data, protocol) if candidate_dir else None
    reasons = []
    if mode == "final":
        require(
            candidate is not None and development_dir and seal_path and seal_sha256,
            "sealed qualification inputs required",
        )
        require_sha(seal_sha256)
        require(kcp13.sha256_of(seal_path) == seal_sha256, "seal digest mismatch")
        seal = read_json(seal_path)
        require(seal["schema"] == "playground-seal-v1", "unsupported seal")
        require(
            seal["implementation_sha256"] == implementation_digest(), "seal implementation mismatch"
        )
        require(seal["benchmark_sha256"] == digest(protocol), "seal protocol mismatch")
        require(seal["candidate_manifest_sha256"] == digest(candidate), "seal candidate mismatch")
        require(seal["final_dataset_sha256"] == digest(data), "seal data mismatch")
        require(data["kind"] == "owner-controlled", "final data must be owner-controlled")
        dev, dev_all = load_dataset(development_dir)
        require(dev["kind"] != "synthetic", "synthetic development cannot qualify a model")
        require(seal["development_dataset_sha256"] == digest(dev), "seal development mismatch")
        check_training_identity(candidate, dev, dev_all)
        # Include excluded draw rows and game/root aliases in the leakage audit.
        for key in ("group_id", "game_id", "root_id"):
            require(
                not (
                    {r[key] for r in all_rows if r.get(key)}
                    & {r[key] for r in dev_all if r.get(key)}
                ),
                "sealed group leakage",
            )
        dev_rows = [r for r in dev_all if r["result"] != 0.5]
        train = [r for r, s in zip(dev_rows, assignments(dev_rows), strict=True) if s == "train"]
        evaluated = rows
        seen = {position_key(r) for r in dev_all}
        split_counts = {"sealed_final": len(rows)}
        overlaps = {"development:final": len(seen & {position_key(r) for r in rows})}
        if overlaps["development:final"]:
            reasons.append("sealed-position-leakage")
    else:
        require(mode == "development", "unknown evaluation mode")
        parts = assignments(rows)
        train = [r for r, s in zip(rows, parts, strict=True) if s == "train"]
        evaluated = [r for r, s in zip(rows, parts, strict=True) if s == "validation"]
        seen = {position_key(r) for r in train}
        split_counts = {s: parts.count(s) for s in ("train", "validation", "test")}
        overlaps = leakage(all_rows, assignments(all_rows))
        seal = None
        if candidate:
            check_training_identity(candidate, data, all_rows)
        reasons.append("development-only")
    require(train and evaluated, "empty training or evaluation split")
    y_train = [r["result"] for r in train]
    prior = float(np.mean(y_train))
    no_info = np.full(len(evaluated), prior)
    x = np.asarray([r["features"] for r in evaluated], dtype=np.float32)
    p = kcp13.predict(Path(candidate_dir) / "model.onnx", x) if candidate else no_info
    reference_manifests = [load_candidate(d, data, protocol) for d in reference_dirs]
    reference_ids = [digest(m) for m in reference_manifests]
    require(len(reference_ids) == len(set(reference_ids)), "duplicate references")
    expected = seal["accepted_references"] if seal else protocol["accepted_references"]
    require(set(reference_ids) == set(expected), "accepted reference inventory mismatch")
    references = {"no-information": no_info}
    for directory, manifest in zip(reference_dirs, reference_manifests, strict=True):
        check_training_identity(
            manifest, dev if mode == "final" else data, dev_all if mode == "final" else all_rows
        )
        references[digest(manifest)] = kcp13.predict(Path(directory) / "model.onnx", x)
    if mode == "final":
        require(seal["promotion_reference"] in references, "unaccepted promotion reference")
        baseline_id = seal["promotion_reference"]
        if expected:
            require(baseline_id != "no-information", "deployable reference required")
    else:
        baseline_id = "no-information"
    baseline = references[baseline_id]
    aggregate = summarize(evaluated, p, baseline, protocol)
    reasons += gate_check(aggregate, protocol)
    unseen_idx = [i for i, r in enumerate(evaluated) if position_key(r) not in seen]
    unseen = None
    if unseen_idx:
        unseen = summarize(
            [evaluated[i] for i in unseen_idx], p[unseen_idx], baseline[unseen_idx], protocol
        )
        reasons += ["unseen:" + reason for reason in gate_check(unseen, protocol)]
    else:
        reasons.append("no-unseen-positions")
    slices = {}
    for name in SLICES:
        idx = [i for i, row in enumerate(evaluated) if name in slice_names(row)]
        if not idx:
            slices[name] = {"status": "unsupported", "reason": "no-applicable-rows"}
            if name != "target:uncertain":
                reasons.append("missing-slice:" + name)
            continue
        summary = summarize([evaluated[i] for i in idx], p[idx], baseline[idx], protocol)
        slices[name] = {"status": "measured", **summary}
        reasons += [name + ":" + r for r in gate_check(summary, protocol, critical_slice=True)]
    probes = probe_report(Path(candidate_dir) / "model.onnx", protocol) if candidate else None
    if probes and not all(probes["checks"].values()):
        reasons.append("golden-probe-failure")
    if mode == "final":
        if evidence_path is None:
            reasons.append("missing-serving-evidence")
        else:
            evidence = read_json(evidence_path)
            reasons += serving_check(evidence, digest(candidate), seal)
    report = {
        "schema": protocol["report_schema"],
        "implementation_sha256": implementation_digest(),
        "benchmark_sha256": digest(protocol),
        "perspective": protocol["perspective"],
        "target": protocol["target"],
        "mode": mode,
        "dataset": {
            "schema": data["schema"],
            "version": data["version"],
            "sha256": digest(data),
            "kind": data["kind"],
        },
        "candidate_manifest_sha256": digest(candidate) if candidate else None,
        "seal_sha256": seal_sha256 if mode == "final" else None,
        "serving_evidence_sha256": digest(read_json(evidence_path))
        if mode == "final" and evidence_path
        else None,
        "counts": {
            "input": len(all_rows),
            "excluded_draws": len(all_rows) - len(rows),
            "splits": split_counts,
            "evaluated_groups": aggregate["confidence"]["groups"],
        },
        "leakage": overlaps,
        "aggregate": aggregate,
        "exact_unseen": unseen,
        "slices": slices,
        "probes": probes,
        "baseline": baseline_id,
        "baselines": {
            name: scores([r["result"] for r in evaluated], prediction)
            for name, prediction in references.items()
        },
        "ineligible_reference_categories": protocol["ineligible_references"],
        "agreement_ranking": {"status": "unsupported", "reason": "no-teacher-or-candidate-lists"},
        "decision": {
            "status": "eligible-for-owner-review" if not reasons else "not-qualified",
            "reasons": sorted(set(reasons)),
        },
    }
    return report
