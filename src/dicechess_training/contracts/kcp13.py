"""The ``kcp-13`` / ``standard-kcp`` serving contract of the Dice Chess evaluation service.

This is the contract the protected evaluation playground currently serves (see
``docs/decisions/0001-playground-train-serve-contract.md``). It pins, on the Python side and
fail-closed, everything the JVM evaluator checks before it mounts a model:

* the feature layout — 13 mover-perspective, **dice-free** floats in ``COLUMN_NAMES`` order,
  produced by the engine's ``KcpFeatures.extract(state, state.activeColor)``;
* the tensor contract — one FLOAT input ``input`` shaped ``[batch, 13]`` and one FLOAT output
  ``output`` shaped ``[batch, 1]``, both with a dynamic batch axis;
* the probability perspective — ``output`` is **P(the side to move wins)**; the evaluator's turn
  analysis scores each candidate afterstate from the *opponent's* (new side to move) view and
  reports ``1 - p`` for the mover;
* the manifest rules — ``manifestVersion`` 1.0.0, matching schema/count/profile, a 64-hex SHA-256
  of the ONNX bytes and an ``engineCompatibility`` comparator set that includes the serving
  engine.

The feature values themselves are deliberately **not** reimplemented in Python: the engine is the
single source of truth (train == serve), and the golden corpus written by
``tools/kcp13-golden`` carries its answers. Only the seven material columns are recomputed here,
as a test oracle for column order — never as a serving path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_ID = "kcp-13"
FEATURE_COUNT = 13
PROFILE_ID = "standard-kcp"
ALGORITHM = "kcp-1ply-onnx"
#: The version this contract writes for a `standard-kcp` position model, and the only one the
#: deployed evaluation service can parse (`SupportedManifestVersion`). Emitting 1.1.0 for that role
#: would make the model unmountable, so readers move first and that writer moves last
#: (fortemate/dicechess-evaluation#88).
MANIFEST_VERSION = "1.0.0"
LEGACY_MANIFEST_VERSION = "1.0.0"
CURRENT_MANIFEST_VERSION = "1.1.0"
SUPPORTED_MANIFEST_VERSIONS: frozenset[str] = frozenset(
    {LEGACY_MANIFEST_VERSION, CURRENT_MANIFEST_VERSION}
)

#: What a model is *used for*. Tensor width cannot tell these apart — two roles may share a feature
#: schema and therefore an identical `[batch, F]` input — so the role is the only thing that can,
#: and feeding one where another is expected produces a bot that runs, logs nothing and plays
#: worse than it measured. The ids are the engine's `ModelRole` ids verbatim.
ROLE_POSITION_VALUE = "position-value"
ROLE_CHANCE_COLLAPSE = "chance-collapse"
ROLE_MOVE_PRERANK = "move-prerank"
SUPPORTED_ROLES: tuple[str, ...] = (
    ROLE_POSITION_VALUE,
    ROLE_CHANCE_COLLAPSE,
    ROLE_MOVE_PRERANK,
)

INPUT_NAME = "input"
OUTPUT_NAME = "output"
MAX_OPSET = 18  # highest default-domain opset the evaluator's ONNX Runtime is verified to load
PERSPECTIVE = "side-to-move"
RULESET_VERSION = "standard-dicechess-v1"

COLUMN_NAMES: tuple[str, ...] = (
    "p_diff",
    "n_diff",
    "b_diff",
    "r_diff",
    "q_diff",
    "material_diff",
    "total_material",
    "mobility_diff",
    "king_safety_diff",
    "king_capture_attack",
    "king_capture_danger",
    "queen_capture_attack",
    "queen_capture_danger",
)
MATERIAL_COLUMNS = COLUMN_NAMES[:7]
CAPTURE_PROBABILITY_COLUMNS = COLUMN_NAMES[9:]

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "kcp13"
GOLDEN_ENGINE_VERSION = "0.9.2"

_PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_COMPARATOR = re.compile(r"^(>=|<=|>|<|=)?(\d+\.\d+\.\d+)$")


class ContractError(ValueError):
    """An artifact, manifest or fixture violates the serving contract. Always fail closed."""


@dataclass(frozen=True)
class GoldenProbe:
    id: str
    fen: str
    side: str
    tags: tuple[str, ...]
    note: str
    features: np.ndarray  # float32, shape (13,)


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
    """Load a JVM-generated golden corpus, rejecting anything that is not this contract."""
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


def twin_fen(fen: str) -> str:
    """The colour-swapped rank-mirror of a position with the side to move flipped.

    A mover-canonical feature vector must be identical for a position and its twin; this is the
    perspective check every golden corpus carries.
    """
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError(f"need at least 4 FEN fields, got {fen!r}")
    placement, side, castling, ep = fields[:4]
    mirrored = "/".join(rank.swapcase() for rank in reversed(placement.split("/")))
    side2 = "b" if side == "w" else "w"
    castling2 = "-" if castling == "-" else "".join(sorted(c.swapcase() for c in castling))
    ep2 = "-" if ep == "-" else ep[0] + str(9 - int(ep[1]))
    return " ".join([mirrored, side2, castling2, ep2, *fields[4:]])


def material_block(fen: str, side: str) -> np.ndarray:
    """The seven material columns from the mover's perspective (own minus opponent).

    A **test oracle only**: it exists to catch a column-order or perspective drift in the golden
    corpus. Serving and training both take their features from the engine.
    """
    if side not in ("w", "b"):
        raise ValueError(f"side must be 'w' or 'b', got {side!r}")
    placement = fen.split()[0]
    white = {piece: placement.count(piece) for piece in "PNBRQ"}
    black = {piece: placement.count(piece.lower()) for piece in "PNBRQ"}
    own, opp = (white, black) if side == "w" else (black, white)
    diffs = [own[piece] - opp[piece] for piece in "PNBRQ"]
    own_material = sum(_PIECE_VALUES[piece] * own[piece] for piece in "PNBRQ")
    opp_material = sum(_PIECE_VALUES[piece] * opp[piece] for piece in "PNBRQ")
    return np.asarray(
        [*diffs, own_material - opp_material, own_material + opp_material], dtype=np.float32
    )


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_version(version: str) -> tuple[int, int, int]:
    match = _VERSION.match(version)
    if not match:
        raise ContractError(f"invalid semantic version {version!r}; expected MAJOR.MINOR.PATCH")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def engine_compatible(constraint: str, engine_version: str) -> bool:
    """Evaluate the evaluator's comparator syntax: space-separated ``>=x.y.z <a.b.c`` with AND
    semantics; a bare version means exact equality."""
    tokens = constraint.split()
    if not tokens:
        raise ContractError("engineCompatibility must not be blank")
    current = _parse_version(engine_version)
    for token in tokens:
        match = _COMPARATOR.match(token)
        if not match:
            raise ContractError(
                f"invalid engineCompatibility {constraint!r}; expected e.g. '>=0.4.0 <0.5.0'"
            )
        operator, version = match.group(1) or "=", _parse_version(match.group(2))
        ok = {
            ">=": current >= version,
            "<=": current <= version,
            ">": current > version,
            "<": current < version,
            "=": current == version,
        }[operator]
        if not ok:
            return False
    return True


def _versioned_field(manifest: dict, name: str, version: str, legacy_default: str) -> str:
    """A field 1.1.0 requires and 1.0.0 never defined.

    Three cases, and the third is the one that matters. Present under 1.1.0 wins. Absent under
    1.0.0 falls back to the legacy default, which is exactly how the engine reads a legacy
    manifest. Present under 1.0.0 is **refused**: every 1.0.0-only reader — this contract until
    now, the deployed evaluation service today — ignores a field its version does not define, so
    honouring it would make one file mean `chance-collapse` here and `position-value` everywhere
    else. A manifest that wants to state a role declares 1.1.0.
    """
    value = manifest.get(name)
    if value is not None:
        if version == LEGACY_MANIFEST_VERSION:
            raise ContractError(f"manifestVersion {version} does not define field {name!r}")
        return str(value)
    if version == LEGACY_MANIFEST_VERSION:
        return legacy_default
    raise ContractError(f"manifestVersion {version} requires field {name!r}")


def manifest_tensor_names(manifest: dict) -> tuple[str, str]:
    """The tensor names the graph must expose: the manifest's, or the contract's defaults.

    Refused under 1.0.0 for the same reason `modelRole` is, and with one extra step. The engine
    reads `inputName` whatever the version, while the evaluation service has no such field and
    reads `input`/`output` — so a 1.0.0 manifest naming its tensors would mean one thing to the
    engine and another to the service. This contract mirrors the service, and more to the point a
    producer should only emit what every consumer reads identically, so it admits the intersection
    and refuses the ambiguity.
    """
    version = manifest.get("manifestVersion")
    names = []
    for key, default in (("inputName", INPUT_NAME), ("outputName", OUTPUT_NAME)):
        value = manifest.get(key)
        if value is None:
            # Absent, or an explicit null: `ManifestFields.optionalString` reads both as absent,
            # and disagreeing with the engine about that would be a divergence of its own.
            names.append(default)
            continue
        if version == LEGACY_MANIFEST_VERSION:
            raise ContractError(f"manifestVersion {version} does not define field {key!r}")
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"{key} must not be blank")
        names.append(value)
    return names[0], names[1]


def validate_manifest(manifest: dict, engine_version: str) -> None:
    """Mirror of the evaluator's ``ModelManifest.validate``: raise ``ContractError`` on the first
    violation instead of loading anything."""
    version = manifest.get("manifestVersion")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        raise ContractError(
            f"unsupported manifestVersion {version!r}; "
            f"expected one of {', '.join(sorted(SUPPORTED_MANIFEST_VERSIONS))}"
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
    if manifest.get("evaluationProfile") != PROFILE_ID:
        raise ContractError(
            f"unsupported evaluationProfile {manifest.get('evaluationProfile')!r}; "
            f"expected {PROFILE_ID!r}"
        )
    role = _versioned_field(manifest, "modelRole", version, ROLE_POSITION_VALUE)
    if role not in SUPPORTED_ROLES:
        raise ContractError(
            f"unknown modelRole {role!r}; known roles: {', '.join(sorted(SUPPORTED_ROLES))}"
        )
    perspective = _versioned_field(manifest, "perspective", version, PERSPECTIVE)
    if perspective != PERSPECTIVE:
        raise ContractError(f"unsupported perspective {perspective!r}; supported: {PERSPECTIVE!r}")
    manifest_tensor_names(manifest)
    constraint = str(manifest.get("engineCompatibility", ""))
    if not engine_compatible(constraint, engine_version):
        raise ContractError(
            f"engineCompatibility {constraint!r} does not include engine version {engine_version!r}"
        )


def verify_model_digest(model_path: str | Path, manifest: dict) -> None:
    actual = sha256_of(model_path)
    if actual.lower() != str(manifest.get("modelSha256", "")).lower():
        raise ContractError(
            f"model SHA-256 mismatch: manifest {manifest.get('modelSha256')}, file {actual}"
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
) -> None:
    """Mirror of the evaluator's ``OnnxModelContract.validate`` on the serialized graph, plus the
    ONNX checker (``onnx.load`` only deserializes) and the opset ceiling from ADR 0001.

    The names default to the contract's own, so a 1.0.0 package validates exactly as before. Pass
    a 1.1.0 manifest's declared names through `manifest_tensor_names`, or a manifest could accept
    tensor names this validator would then reject.
    """
    import onnx

    model = onnx.load(str(model_path))
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as err:  # onnx raises ValidationError or a shape-inference error
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
    """Run a contract-conforming ONNX model on ``[n, 13]`` rows the way the evaluator does:
    FLOAT in, one finite value per row out, clamped to ``[0, 1]``. Non-finite output fails."""
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
    model_role: str = ROLE_POSITION_VALUE,
    input_name: str = INPUT_NAME,
    output_name: str = OUTPUT_NAME,
) -> dict:
    """Assemble a manifest for a contract-conforming ONNX artifact. The digest is computed from the
    bytes on disk so the manifest can only ever describe the exact file it was built from.

    The role decides the version, and it is not a preference. A position model keeps writing 1.0.0
    because that is the only version the deployed evaluation service can parse, and a 1.1.0
    position model would simply not mount; any other role can only be described by 1.1.0, since
    1.0.0 has no field to say what it is. Writing 1.1.0 for position models becomes safe once the
    service reads it, and is a separate change on purpose.
    """
    if model_role not in SUPPORTED_ROLES:
        raise ContractError(
            f"unknown modelRole {model_role!r}; known roles: {', '.join(sorted(SUPPORTED_ROLES))}"
        )
    # A position model is always 1.0.0, with no escape hatch. 1.0.0 cannot name its tensors, and
    # emitting 1.1.0 to say so would produce a package the deployed service cannot parse and
    # therefore cannot mount — an artifact that describes itself perfectly and serves nowhere. So
    # renamed tensors are refused for this role until the service reads 1.1.0, rather than
    # quietly produced in a form nothing can load.
    if model_role == ROLE_POSITION_VALUE and (input_name, output_name) != (
        INPUT_NAME,
        OUTPUT_NAME,
    ):
        raise ContractError(
            f"a {ROLE_POSITION_VALUE} artifact cannot rename its tensors: "
            f"{LEGACY_MANIFEST_VERSION} has no field for it and the evaluation service reads "
            f"no other version"
        )
    version = (
        LEGACY_MANIFEST_VERSION if model_role == ROLE_POSITION_VALUE else CURRENT_MANIFEST_VERSION
    )
    manifest = {
        "manifestVersion": version,
        "modelId": model_id,
        "modelSha256": sha256_of(model_path),
        "featureSchema": SCHEMA_ID,
        "featureCount": FEATURE_COUNT,
        "engineCompatibility": engine_compatibility,
        "evaluationProfile": PROFILE_ID,
        "calibration": {"temperature": 1.0, **(calibration or {})},
        "provenance": dict(provenance or {}),
    }
    if version == CURRENT_MANIFEST_VERSION:
        manifest["modelRole"] = model_role
        manifest["perspective"] = PERSPECTIVE
        manifest["inputName"] = input_name
        manifest["outputName"] = output_name
    return manifest
