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
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dicechess_training.ablation.export import export_candidate
from dicechess_training.ablation.runner import predict as torch_predict
from dicechess_training.ablation.runner import select_epoch_budget, train_model
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


def build_candidate(
    dataset_dir: str | Path,
    output_dir: str | Path,
    config: CandidateConfig,
) -> dict[str, Any]:
    """Train and package a candidate, or raise and write nothing."""
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

    identity = benchmark_core.train_identity(data_manifest, rows)
    provenance = {
        **identity,
        "config_sha256": benchmark_core.digest(config.as_record()),
        "benchmark_sha256": benchmark_core.digest(protocol),
        "code_sha256": _code_digest(),
        "engine_version": data_manifest["engine_version"],
        "seed": int(config.seed),
        "perspective": "side-to-move",
        "selected_epochs": int(selection["selected_epochs"]),
        "feature_schema": kcp13.SCHEMA_ID,
    }

    with tempfile.TemporaryDirectory() as staging:
        staged_model = Path(staging) / MODEL_FILE
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
            "training_rows": len(decisive),
            "inner_tuning_rows": len(inner),
            "epoch_selection": selection,
            "dataset_version": data_manifest["version"],
        }

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name in (MODEL_FILE, MANIFEST_FILE, MODEL_CARD_FILE):
            _require(not (destination / name).exists(), f"{name} already exists in the output")
        shutil.move(str(staged_model), destination / MODEL_FILE)
        (destination / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (destination / MODEL_CARD_FILE).write_text(_model_card(summary), encoding="utf-8")

    return summary
