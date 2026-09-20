"""The artifact contract for the move pre-ranker: `rich-9-v1` features, role `move-prerank`.

A pre-ranker is not a value model with different weights, and four differences make copying the
evaluator's contract the wrong move.

**Role.** The engine keeps three model seams and refuses a package whose declared role is not the
one the seam wants. Ordering candidates and valuing positions are different jobs with different
targets; that both emit one number per row is not a reason to let one stand in for the other.

**Manifest version.** `modelRole` and `perspective` exist only from 1.1.0, so an artifact that needs
a role must declare 1.1.0. The evaluator still writes 1.0.0 because the deployed service pins it —
the pre-ranker is a new role, and new roles are where 1.1.0 starts.

**The output is a score, not a probability.** An ordering is invariant to monotone transforms, so
there is nothing to calibrate and nothing to clamp. Clipping a ranker's output into [0, 1] — which
the evaluator's contract does, correctly, for a win probability — would collapse exactly the
distinctions this model exists to make. A `calibration` block is therefore refused rather than
ignored: it would mean the producer thought this was a value model.

**Cost, which chooses the feature schema.** The seam scores every legal turn before the search's
deadline is consulted, so the schema has to be one whose cost does not follow the branching factor.
That is `rich-9-v1` here, and the column layout is read from the committed golden corpus rather
than restated, so this file cannot invent a feature.

Deliberately not registered in `SCHEMA_CONTRACTS`: that mapping answers "which value-model contract
reads this enriched shard and exports under these names", and a pre-ranker is neither.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .kcp13 import ContractError, engine_compatible, sha256_of

SCHEMA_ID = "rich-9-v1"
ROLE = "move-prerank"
MANIFEST_VERSION = "1.1.0"
PERSPECTIVE = "side-to-move"
INPUT_NAME = "input"
OUTPUT_NAME = "output"
MAX_OPSET = 18
DEFAULT_ENGINE_VERSION = "0.12.0"

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "rich9"

#: Matched rather than parsed: a 64-character string that is not hexadecimal must be refused the
#: same way as every other malformed field, and `int(value, 16)` would raise a ValueError past
#: the callers that only catch ContractError.
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def golden_path(engine_version: str = DEFAULT_ENGINE_VERSION) -> Path:
    return GOLDEN_DIR / f"golden-engine-{engine_version}.json"


def columns(engine_version: str = DEFAULT_ENGINE_VERSION) -> tuple[str, ...]:
    """The engine's column layout for this schema, read from its own committed answer."""
    path = golden_path(engine_version)
    if not path.is_file():
        raise ContractError(f"no committed golden corpus for engine {engine_version!r}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA_ID:
        raise ContractError(f"{path}: schema {raw.get('schema')!r} != {SCHEMA_ID!r}")
    layout = tuple(raw.get("columns", ()))
    if not layout:
        raise ContractError(f"{path}: golden corpus declares no columns")
    return layout


def feature_count(engine_version: str = DEFAULT_ENGINE_VERSION) -> int:
    return len(columns(engine_version))


def build_manifest(
    model_path: str | Path,
    model_id: str,
    engine_compatibility: str,
    provenance: dict[str, str] | None = None,
    engine_version: str = DEFAULT_ENGINE_VERSION,
    input_name: str = INPUT_NAME,
    output_name: str = OUTPUT_NAME,
) -> dict:
    """Assemble a 1.1.0 manifest for a pre-ranking artifact.

    `inputName` and `outputName` are written explicitly even when they are the defaults: 1.1.0 has
    fields for them, and a consumer that opens the session should not have to guess from a
    convention this repository happens to follow.
    """
    return {
        "manifestVersion": MANIFEST_VERSION,
        "modelId": model_id,
        "modelSha256": sha256_of(model_path),
        "modelRole": ROLE,
        "perspective": PERSPECTIVE,
        "featureSchema": SCHEMA_ID,
        "featureCount": feature_count(engine_version),
        "engineCompatibility": engine_compatibility,
        "inputName": input_name,
        "outputName": output_name,
        "provenance": dict(provenance or {}),
    }


def manifest_tensor_names(manifest: dict) -> tuple[str, str]:
    """The names the graph must expose, as the manifest declares them."""
    return (
        str(manifest.get("inputName", INPUT_NAME)),
        str(manifest.get("outputName", OUTPUT_NAME)),
    )


def validate_manifest(manifest: dict, engine_version: str) -> None:
    version = manifest.get("manifestVersion")
    if version != MANIFEST_VERSION:
        raise ContractError(
            f"unsupported manifestVersion {version!r}; a pre-ranker declares a role and "
            f"{MANIFEST_VERSION!r} is the first version that has one"
        )
    if manifest.get("modelRole") != ROLE:
        raise ContractError(f"modelRole {manifest.get('modelRole')!r} cannot serve as {ROLE!r}")
    if manifest.get("perspective") != PERSPECTIVE:
        raise ContractError(
            f"unsupported perspective {manifest.get('perspective')!r}; expected {PERSPECTIVE!r}"
        )
    if not str(manifest.get("modelId", "")).strip():
        raise ContractError("modelId must not be blank")
    if not _SHA256.match(str(manifest.get("modelSha256", ""))):
        raise ContractError("modelSha256 must be exactly 64 hexadecimal characters")
    if manifest.get("featureSchema") != SCHEMA_ID:
        raise ContractError(
            f"unsupported featureSchema {manifest.get('featureSchema')!r}; expected {SCHEMA_ID!r}"
        )
    expected = feature_count(engine_version)
    if manifest.get("featureCount") != expected:
        raise ContractError(
            f"unsupported featureCount {manifest.get('featureCount')!r}; expected {expected}"
        )
    # Present-and-wrong is the failure this catches: a producer that filled in a calibration block
    # was treating a ranking score as a probability, and the ordering it learned means something
    # else than it thinks.
    if "calibration" in manifest:
        raise ContractError(
            "a ranking score has nothing to calibrate; remove the calibration block"
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


def validate_onnx_contract(
    model_path: str | Path,
    input_name: str = INPUT_NAME,
    output_name: str = OUTPUT_NAME,
    engine_version: str = DEFAULT_ENGINE_VERSION,
) -> None:
    """Check the graph the engine will open: names, dtypes, shapes, opset.

    The batch axis must stay dynamic. The engine feeds this model every legal turn in bounded
    chunks, so a graph pinned to one row count would refuse the only workload it has.
    """
    import onnx

    width = feature_count(engine_version)
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
    if len(inputs) != 1 or inputs[0].name != input_name:
        raise ContractError(
            f"ONNX model must expose exactly one input named {input_name!r}; "
            f"found {[t.name for t in inputs]}"
        )
    if len(outputs) != 1 or outputs[0].name != output_name:
        raise ContractError(
            f"ONNX model must expose exactly one output named {output_name!r}; "
            f"found {[t.name for t in outputs]}"
        )
    float_type = onnx.TensorProto.FLOAT
    if inputs[0].type.tensor_type.elem_type != float_type:
        raise ContractError(f"input {input_name!r} must have FLOAT dtype")
    if outputs[0].type.tensor_type.elem_type != float_type:
        raise ContractError(f"output {output_name!r} must have FLOAT dtype")
    in_dims, out_dims = _dims(inputs[0]), _dims(outputs[0])
    if len(in_dims) != 2 or in_dims[0] is not None or in_dims[1] != width:
        raise ContractError(
            f"input {input_name!r} must have shape [batch,{width}] with a dynamic batch; "
            f"got {in_dims}"
        )
    if len(out_dims) != 2 or out_dims[0] is not None or out_dims[1] != 1:
        raise ContractError(
            f"output {output_name!r} must have shape [batch,1] with a dynamic batch; got {out_dims}"
        )


def score(
    model_path: str | Path,
    features: np.ndarray,
    input_name: str = INPUT_NAME,
    output_name: str = OUTPUT_NAME,
    engine_version: str = DEFAULT_ENGINE_VERSION,
) -> np.ndarray:
    """Run the ranker and return its scores, unclamped.

    No clipping, and that is the point. The evaluator's contract clamps to [0, 1] because its
    output is a probability; here the number is only meaningful against the other candidates of the
    same root, and squeezing it into an interval would erase differences at both ends. Finiteness
    is still enforced — a NaN has no place in an ordering.
    """
    import onnxruntime as ort

    width = feature_count(engine_version)
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != width:
        raise ContractError(f"features must be [n,{width}], got {x.shape}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (output,) = session.run([output_name], {input_name: x})
    output = np.asarray(output)
    if output.shape != (len(x), 1):
        raise ContractError(f"ONNX output has shape {output.shape}; expected {(len(x), 1)}")
    scores = output.reshape(-1).astype(np.float64)
    if not all(math.isfinite(value) for value in scores):
        raise ContractError("ONNX output contains a non-finite score")
    return scores
