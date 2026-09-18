"""Train a `kcp-13` candidate and package it for the frozen benchmark.

Three properties matter more than the training itself. The dataset is admitted by the
benchmark's own loader before a single weight is fitted, so a corpus the benchmark would
reject never produces an artifact. The split, the inner tuning cutoff and the digests come
from the benchmark's own helpers rather than from a second implementation, so the package
cannot drift from its consumer. And nothing is written until the finished package passes the
checks the benchmark will apply to it, so a failed run leaves no half-valid artifact behind
for someone to find later and trust.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dicechess_training.ablation.export import export_candidate
from dicechess_training.ablation.runner import (
    TEMPERATURE_RULES,
    fit_logit_temperature,
    select_epoch_budget,
    train_model,
)
from dicechess_training.ablation.runner import predict as torch_predict
from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.benchmark.splits import assignments
from dicechess_training.contracts import kcp13

MODEL_FILE = "model.onnx"
MANIFEST_FILE = "manifest.json"
MODEL_CARD_FILE = "model-card.md"

#: ADR 0001, amended by Issue #30: the range whose extraction the golden corpus replays.
ENGINE_COMPATIBILITY = ">=0.4.0 <0.13.0"

#: ADR 0001 Decision 6a: Torch versus onnxruntime on the golden matrix and held-out rows.
PARITY_TOLERANCE = 1e-6

ROOT = Path(__file__).resolve().parents[3]


class CandidateError(ValueError):
    """A candidate could not be produced. Messages never carry dataset paths or row content."""


@dataclass(frozen=True)
class CandidateConfig:
    """Everything that changes the weights, and nothing that does not.

    The whole record is digested into `config_sha256`, so a rerun that differs in any field
    is a different candidate by construction.
    """

    seed: int
    model_id: str
    hidden_dims: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    batch_size: int = 256
    loss: str = "bce-with-logits"
    feature_standardisation: str = "train-statistics"
    epoch_candidates: tuple[int, ...] = (5, 10, 20, 40)
    inner_cutoff: int = 7000
    engine_compatibility: str = ENGINE_COMPATIBILITY
    calibration: dict[str, Any] = field(default_factory=dict)
    #: Scoring rule the graph-embedded probability calibration is selected by, or `none` to ship
    #: the raw sigmoid. Declared here rather than chosen after the fact, so it is digested into
    #: `config_sha256` before the run and a candidate cannot acquire its calibration rule by
    #: looking at how it scored.
    probability_calibration: str = "brier"

    def training_config(self) -> dict[str, Any]:
        return {
            "hidden_dims": list(self.hidden_dims),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "loss": self.loss,
            "feature_standardisation": self.feature_standardisation,
            "epoch_selection": {"candidates": list(self.epoch_candidates)},
        }

    def as_record(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "model_id": self.model_id,
            "hidden_dims": list(self.hidden_dims),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "loss": self.loss,
            "feature_standardisation": self.feature_standardisation,
            "epoch_candidates": list(self.epoch_candidates),
            "inner_cutoff": self.inner_cutoff,
            "engine_compatibility": self.engine_compatibility,
            "probability_calibration": self.probability_calibration,
        }


def group_value(group: str) -> int:
    """The split hash of `benchmark.splits.split_for`, as a number.

    The benchmark exposes only the partition name, and the inner tuning cutoff needs the value
    underneath it. `test_candidate.py` asserts this agrees with `split_for` so the two cannot
    drift apart.
    """
    return int(hashlib.sha256(("playground-v1:" + group).encode()).hexdigest(), 16) % 10000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateError(message)


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["features"] for row in rows], dtype=np.float32)
    y = np.asarray([row["result"] for row in rows], dtype=np.float32)
    return x, y


def _code_digest() -> str:
    """Digest of the code that decides the weights, plus the resolved dependency set."""
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [
        ROOT / "src/dicechess_training/ablation/runner.py",
        ROOT / "src/dicechess_training/ablation/export.py",
        Path(kcp13.__file__),
        ROOT / "uv.lock",
    ]
    return benchmark_core.digest({str(p.relative_to(ROOT)): kcp13.sha256_of(p) for p in paths})


def _parity_rows(rows: list[dict], parts: list[str]) -> np.ndarray:
    """Held-out rows for the parity check: validation if the dataset has any, else train.

    Parity is a numerical identity between two runtimes, not a selection signal, so reading
    the validation partition here buys the candidate nothing. The final test partition is
    still not touched.
    """
    held_out = [row for row, part in zip(rows, parts, strict=True) if part == "validation"]
    source = held_out or [row for row, part in zip(rows, parts, strict=True) if part == "train"]
    return np.asarray([row["features"] for row in source], dtype=np.float32)


def _check_parity(model, model_path: Path, held_out: np.ndarray) -> dict[str, float]:
    golden = kcp13.load_golden().matrix()
    report = {}
    for name, matrix in (("golden", golden), ("held_out", held_out)):
        if len(matrix) == 0:  # pragma: no cover - datasets always have train rows
            continue
        torch_p = torch_predict(model, matrix).astype(np.float64)
        onnx_p = kcp13.predict(model_path, matrix).astype(np.float64)
        diff = float(np.max(np.abs(torch_p - onnx_p)))
        _require(
            diff <= PARITY_TOLERANCE,
            f"{name} parity {diff:.3e} exceeds the contract tolerance {PARITY_TOLERANCE:.0e}",
        )
        report[f"{name}_max_abs_diff"] = diff
    return report


def _model_card(summary: dict[str, Any]) -> str:
    template = (ROOT / "docs/benchmark/model-card-template.md").read_text(encoding="utf-8")
    provenance = summary["provenance"]
    filled = "\n".join(
        [
            "",
            "## Filled by the packager",
            "",
            "These rows are read from the artifacts. Every other row above stays the owner's to",
            "complete in the private audit store; nothing here identifies data, hosts or verdicts.",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Feature schema | `{kcp13.SCHEMA_ID}` ({kcp13.FEATURE_COUNT} columns) |",
            f"| Perspective | `{provenance['perspective']}` |",
            f"| Enrichment engine | `{provenance['engine_version']}` |",
            f"| Engine compatibility | `{summary['manifest']['engineCompatibility']}` |",
            f"| Training seed | `{provenance['seed']}` |",
            f"| Selected epochs | `{provenance['selected_epochs']}` |",
            f"| Probability calibration | `{provenance['probability_calibration']}`, "
            f"embedded in the graph |",
            f"| Logit temperature | `{provenance['logit_temperature']}` |",
            f"| Training rows (decisive) | `{summary['training_rows']}` |",
            f"| Model digest | `{summary['manifest']['modelSha256']}` |",
            f"| Training data digest | `{provenance['training_data_sha256']}` |",
            f"| Training groups digest | `{provenance['training_groups_sha256']}` |",
            f"| Config digest | `{provenance['config_sha256']}` |",
            f"| Benchmark protocol digest | `{provenance['benchmark_sha256']}` |",
            f"| Code digest | `{provenance['code_sha256']}` |",
            f"| Torch/ONNX parity (golden) | `{summary['parity']['golden_max_abs_diff']:.3e}` |",
            "",
            "Qualification and deployment decisions are not recorded by this file: the benchmark",
            "judges the package, and promotion stays an owner decision.",
            "",
        ]
    )
    return template.rstrip("\n") + "\n" + filled


@dataclass(frozen=True)
class TrainedCandidate:
    """Everything the packager needs, and everything a later audit needs to recover the model.

    The serving evidence has to compare the manifest-bound training model against the exported
    bytes, and the package is fixed at three files, so the model cannot travel with it. It is
    recovered by running this again — which is the same thing the packager runs, so the audit
    cannot drift from what was shipped.
    """

    model: Any
    provenance: dict[str, str]
    selection: dict[str, Any]
    calibration: dict[str, Any] | None
    data_manifest: dict[str, Any]
    rows: list[dict]
    parts: list[str]
    training_rows: int
    inner_tuning_rows: int


def train_candidate(dataset_dir: str | Path, config: CandidateConfig) -> TrainedCandidate:
    """Admit the dataset, train, calibrate and derive provenance. Writes nothing."""
    _require(
        config.probability_calibration in ("none", *TEMPERATURE_RULES),
        "unsupported probability calibration rule",
    )
    data_manifest, rows = benchmark_core.load_dataset(dataset_dir)
    protocol = benchmark_core.load_protocol()

    parts = assignments(rows)
    train_rows = [row for row, part in zip(rows, parts, strict=True) if part == "train"]
    decisive = [row for row in train_rows if row["result"] in (0, 1)]
    _require(bool(decisive), "the training partition holds no decisive rows")

    fit = [row for row in decisive if group_value(row["group_id"]) < config.inner_cutoff]
    inner = [row for row in decisive if group_value(row["group_id"]) >= config.inner_cutoff]
    _require(bool(fit), "the inner fit split is empty; lower inner_cutoff or add training groups")
    _require(bool(inner), "the inner tuning split is empty; raise inner_cutoff or add groups")

    training_config = config.training_config()
    x_fit, y_fit = _matrix(fit)
    x_inner, y_inner = _matrix(inner)
    selection = select_epoch_budget(
        x_fit, y_fit, x_inner, y_inner, kcp13.FEATURE_COUNT, training_config, config.seed
    )

    x_train, y_train = _matrix(decisive)
    model = train_model(
        x_train,
        y_train,
        kcp13.FEATURE_COUNT,
        training_config,
        config.seed,
        epochs=selection["selected_epochs"],
    )

    # The calibration is fitted against a model that has never seen the inner tuning split, at the
    # budget already selected on it, and the constant is then carried to the shipped model. The
    # shipped model is refit on fit+inner, so the inner split is not held out for it; a constant
    # fitted there would read that model's memorised confidence as calibration and under-correct.
    # Both models share architecture, seed and budget, and differ only by the ~12% of training
    # rows added by the refit, which is the same transfer assumption the epoch budget already makes.
    calibration = None
    if config.probability_calibration != "none":
        probe = train_model(
            x_fit,
            y_fit,
            kcp13.FEATURE_COUNT,
            training_config,
            config.seed,
            epochs=selection["selected_epochs"],
        )
        calibration = fit_logit_temperature(probe, x_inner, y_inner, config.probability_calibration)
        model.set_logit_temperature(calibration["temperature"])

    identity = benchmark_core.train_identity(data_manifest, rows)
    provenance = {
        **identity,
        "config_sha256": benchmark_core.digest(config.as_record()),
        "benchmark_sha256": benchmark_core.digest(protocol),
        "code_sha256": _code_digest(),
        "engine_version": data_manifest["engine_version"],
        "seed": str(config.seed),
        "perspective": "side-to-move",
        "selected_epochs": str(selection["selected_epochs"]),
        "feature_schema": kcp13.SCHEMA_ID,
        "probability_calibration": config.probability_calibration,
        "logit_temperature": repr(calibration["temperature"]) if calibration else "1.0",
    }

    return TrainedCandidate(
        model=model,
        provenance=provenance,
        selection=selection,
        calibration=calibration,
        data_manifest=data_manifest,
        rows=rows,
        parts=parts,
        training_rows=len(decisive),
        inner_tuning_rows=len(inner),
    )


def build_candidate(
    dataset_dir: str | Path,
    output_dir: str | Path,
    config: CandidateConfig,
) -> dict[str, Any]:
    """Train and package a candidate, or raise and write nothing."""
    trained = train_candidate(dataset_dir, config)
    model = trained.model
    provenance = trained.provenance
    calibration = trained.calibration
    selection = trained.selection
    data_manifest = trained.data_manifest
    rows = trained.rows
    parts = trained.parts

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (MODEL_FILE, MANIFEST_FILE, MODEL_CARD_FILE):
        _require(not (destination / name).exists(), f"{name} already exists in the output")

    # Staged beside the destination rather than in the system temp directory, so publication is
    # a same-filesystem rename per file and a failed build cannot leave a half-written package.
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".candidate-staging-"))
    try:
        staged_model = staging / MODEL_FILE
        export_candidate(model, kcp13.SCHEMA_ID, staged_model)
        manifest = kcp13.build_manifest(
            staged_model,
            config.model_id,
            config.engine_compatibility,
            calibration=config.calibration or None,
            provenance=provenance,
        )

        # Exactly what the benchmark will do to this package, before it is allowed to exist.
        kcp13.validate_manifest(manifest, data_manifest["engine_version"])
        kcp13.verify_model_digest(staged_model, manifest)
        kcp13.validate_onnx_contract(staged_model)
        benchmark_core.check_training_identity(manifest, data_manifest, rows)
        _require(
            manifest.get("calibration", {}) in ({}, {"temperature": 1.0}),
            "calibration must be embedded in the graph, not declared in the manifest",
        )
        parity = _check_parity(model, staged_model, _parity_rows(rows, parts))

        summary = {
            "model_id": config.model_id,
            "manifest": manifest,
            "provenance": provenance,
            "parity": parity,
            "training_rows": trained.training_rows,
            "inner_tuning_rows": trained.inner_tuning_rows,
            "epoch_selection": selection,
            "calibration": calibration,
            "dataset_version": data_manifest["version"],
        }

        (staging / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / MODEL_CARD_FILE).write_text(_model_card(summary), encoding="utf-8")

        published: list[Path] = []
        try:
            for name in (MODEL_FILE, MANIFEST_FILE, MODEL_CARD_FILE):
                os.replace(staging / name, destination / name)
                published.append(destination / name)
        except OSError:
            # A package missing one of its three files is worse than no package at all.
            for path in published:
                path.unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return summary
