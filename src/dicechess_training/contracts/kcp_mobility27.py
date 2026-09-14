"""The ``kcp-mobility-27-v1`` contract for candidate schema S1.

Extends the baseline ``kcp-13`` contract with 12 pseudo-legal move counts per moving piece
type (own and opponent) and 2 normalized piece diversity indices (own and opponent PDI).

Column layout:
* 13 ``kcp`` columns (exact prefix identity to ``kcp-13``);
* 6 own move counts: ``own_moves_p``, ``own_moves_n``, ``own_moves_b``,
  ``own_moves_r``, ``own_moves_q``, ``own_moves_k``;
* 6 opp move counts: ``opp_moves_p``, ``opp_moves_n``, ``opp_moves_b``,
  ``opp_moves_r``, ``opp_moves_q``, ``opp_moves_k``;
* 2 PDI columns: ``own_pdi``, ``opp_pdi`` (in 0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

All features evaluated from the side-to-move perspective.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kcp13 import (
    COLUMN_NAMES as KCP13_COLUMNS,
)
from .kcp13 import (
    ContractError,
    engine_compatible,
    sha256_of,
)

SCHEMA_ID = "kcp-mobility-27-v1"
FEATURE_COUNT = 27
PROFILE_ID = "standard-kcp-mobility-27"
MANIFEST_VERSION = "1.0.0"
INPUT_NAME = "input"
OUTPUT_NAME = "output"
MAX_OPSET = 18
PERSPECTIVE = "side-to-move"

MOBILITY_OWN_COLUMNS: tuple[str, ...] = (
    "own_moves_p",
    "own_moves_n",
    "own_moves_b",
    "own_moves_r",
    "own_moves_q",
    "own_moves_k",
)
MOBILITY_OPP_COLUMNS: tuple[str, ...] = (
    "opp_moves_p",
    "opp_moves_n",
    "opp_moves_b",
    "opp_moves_r",
    "opp_moves_q",
    "opp_moves_k",
)
PDI_COLUMNS: tuple[str, ...] = ("own_pdi", "opp_pdi")

COLUMN_NAMES: tuple[str, ...] = (
    *KCP13_COLUMNS,
    *MOBILITY_OWN_COLUMNS,
    *MOBILITY_OPP_COLUMNS,
    *PDI_COLUMNS,
)

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "kcp_mobility27"
GOLDEN_ENGINE_VERSION = "0.9.3"

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class GoldenProbe:
    id: str
    fen: str
    side: str
    tags: tuple[str, ...]
    note: str
    features: np.ndarray  # float32, shape (27,)


@dataclass(frozen=True)
class GoldenCorpus:
    schema: str
    perspective: str
    engine_version: str
    engine_artifact: str
    columns: tuple[str, ...]
    probes: tuple[GoldenProbe, ...]

    def by_id(self) -> dict[str, GoldenProbe]:
        return {probe.id: probe for probe in self.probes}

    def matrix(self) -> np.ndarray:
        return np.stack([probe.features for probe in self.probes])


def golden_path(engine_version: str = GOLDEN_ENGINE_VERSION) -> Path:
    return GOLDEN_DIR / f"golden-engine-{engine_version}.json"


def load_golden(path: str | Path | None = None) -> GoldenCorpus:
    """Load a JVM-generated golden corpus for kcp-mobility-27-v1."""
    resolved = Path(path) if path is not None else golden_path()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA_ID:
        raise ContractError(f"{resolved}: schema {raw.get('schema')!r} != {SCHEMA_ID!r}")
    if raw.get("perspective") != PERSPECTIVE:
        raise ContractError(
            f"{resolved}: perspective {raw.get('perspective')!r} != {PERSPECTIVE!r}"
        )
    columns = tuple(raw.get("columns", ()))
    if columns != COLUMN_NAMES:
        raise ContractError(f"{resolved}: column layout {columns} != contract {COLUMN_NAMES}")
    probes = []
    for entry in raw.get("probes", ()):
        features = np.asarray(entry["features"], dtype=np.float32)
        if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
            raise ContractError(f"{resolved}: probe {entry.get('id')!r}: invalid feature vector")
        if entry["side"] not in ("w", "b"):
            raise ContractError(f"{resolved}: probe {entry.get('id')!r} has side {entry['side']!r}")
        probes.append(
            GoldenProbe(
                id=entry["id"],
                fen=entry["fen"],
                side=entry["side"],
                tags=tuple(entry.get("tags", ())),
                note=entry.get("note", ""),
                features=features,
            )
        )
    if not probes:
        raise ContractError(f"{resolved}: no probes")
    ids = [probe.id for probe in probes]
    if len(set(ids)) != len(ids):
        raise ContractError(f"{resolved}: duplicate probe ids")
    return GoldenCorpus(
        schema=raw["schema"],
        perspective=raw["perspective"],
        engine_version=raw["engineVersion"],
        engine_artifact=raw["engineArtifact"],
        columns=columns,
        probes=tuple(probes),
    )


def validate_manifest(manifest: dict, engine_version: str) -> None:
    if manifest.get("manifestVersion") != MANIFEST_VERSION:
        raise ContractError(
            f"unsupported manifestVersion {manifest.get('manifestVersion')!r}; "
            f"expected {MANIFEST_VERSION!r}"
        )
    if not str(manifest.get("modelId", "")).strip():
        raise ContractError("modelId must not be blank")
    if not _SHA256.match(str(manifest.get("modelSha256", ""))):
        raise ContractError("modelSha256 must be exactly 64 hexadecimal characters")
    if manifest.get("featureSchema") != SCHEMA_ID:
        raise ContractError(
            f"unsupported featureSchema {manifest.get('featureSchema')!r}; expected {SCHEMA_ID!r}"
        )
    if manifest.get("featureCount") != FEATURE_COUNT:
        raise ContractError(
            f"unsupported featureCount {manifest.get('featureCount')!r}; expected {FEATURE_COUNT}"
        )
    constraint = str(manifest.get("engineCompatibility", ""))
    if not engine_compatible(constraint, engine_version):
        raise ContractError(
            f"engineCompatibility {constraint!r} does not include engine version {engine_version!r}"
        )


def _dims(tensor) -> list[int | None]:
    return [
        None if dim.dim_param or dim.dim_value == 0 else dim.dim_value
        for dim in tensor.type.tensor_type.shape.dim
    ]


def validate_onnx_contract(model_path: str | Path) -> None:
    import onnx

    model = onnx.load(str(model_path))
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as err:
        raise ContractError(f"ONNX model is not a valid graph: {err}") from err
    default_opsets = [
        entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")
    ]
    if not default_opsets:
        raise ContractError("ONNX model declares no default-domain opset")
    if max(default_opsets) > MAX_OPSET:
        raise ContractError(
            f"ONNX default-domain opset {max(default_opsets)} exceeds the maximum {MAX_OPSET}"
        )
    initializers = {init.name for init in model.graph.initializer}
    inputs = [tensor for tensor in model.graph.input if tensor.name not in initializers]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or inputs[0].name != INPUT_NAME:
        raise ContractError(
            f"ONNX model must expose exactly one input named {INPUT_NAME!r}; "
            f"found {[t.name for t in inputs]}"
        )
    if len(outputs) != 1 or outputs[0].name != OUTPUT_NAME:
        raise ContractError(
            f"ONNX model must expose exactly one output named {OUTPUT_NAME!r}; "
            f"found {[t.name for t in outputs]}"
        )
    float_type = onnx.TensorProto.FLOAT
    if inputs[0].type.tensor_type.elem_type != float_type:
        raise ContractError(f"input {INPUT_NAME!r} must have FLOAT dtype")
    if outputs[0].type.tensor_type.elem_type != float_type:
        raise ContractError(f"output {OUTPUT_NAME!r} must have FLOAT dtype")
    in_dims, out_dims = _dims(inputs[0]), _dims(outputs[0])
    if len(in_dims) != 2 or in_dims[0] is not None or in_dims[1] != FEATURE_COUNT:
        raise ContractError(
            f"input {INPUT_NAME!r} must have shape [batch,{FEATURE_COUNT}] with a dynamic batch; "
            f"got {in_dims}"
        )
    if len(out_dims) != 2 or out_dims[0] is not None or out_dims[1] != 1:
        raise ContractError(
            f"output {OUTPUT_NAME!r} must have shape [batch,1] with a dynamic batch; got {out_dims}"
        )


def predict(model_path: str | Path, features: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != FEATURE_COUNT:
        raise ContractError(f"features must be [n,{FEATURE_COUNT}], got {x.shape}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (output,) = session.run([OUTPUT_NAME], {INPUT_NAME: x})
    output = np.asarray(output)
    if output.shape != (len(x), 1):
        raise ContractError(f"ONNX output has shape {output.shape}; expected {(len(x), 1)}")
    if not np.isfinite(output).all():
        raise ContractError("ONNX output contains a non-finite value")
    return np.clip(output.reshape(-1).astype(np.float64), 0.0, 1.0)


def build_manifest(
    model_path: str | Path,
    model_id: str,
    engine_compatibility: str,
    calibration: dict | None = None,
    provenance: dict[str, str] | None = None,
) -> dict:
    return {
        "manifestVersion": MANIFEST_VERSION,
        "modelId": model_id,
        "modelSha256": sha256_of(model_path),
        "featureSchema": SCHEMA_ID,
        "featureCount": FEATURE_COUNT,
        "engineCompatibility": engine_compatibility,
        "evaluationProfile": PROFILE_ID,
        "calibration": {"temperature": 1.0, **(calibration or {})},
        "provenance": dict(provenance or {}),
    }
