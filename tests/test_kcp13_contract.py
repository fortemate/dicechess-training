"""The `kcp-13` serving contract: golden-corpus invariants and fail-closed validators.

The golden vectors come from the engine (tools/kcp13-golden); these tests never recompute the
positional or capture-probability columns. They check what a contract mismatch would break:
column order, mover-canonical perspective, canonicalization, bounds, and the manifest/ONNX rules
the evaluator enforces before it mounts a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from dicechess_training.contracts import kcp13

# The three dice are piece-type dice, so a lone rook (or pawn) capture needs its own face:
# P(at least one specific face among three) = 1 - (5/6)^3 = 91/216. A capture that needs two
# queen moves needs at least two Queen faces: 3 * (1/6)^2 * (5/6) + (1/6)^3 = 16/216.
ONE_FACE = np.float32(91 / 216)
TWO_QUEEN_FACES = np.float32(16 / 216)


@pytest.fixture(scope="module")
def golden() -> kcp13.GoldenCorpus:
    return kcp13.load_golden()


@pytest.fixture(scope="module")
def probes(golden) -> dict[str, kcp13.GoldenProbe]:
    return golden.by_id()


def col(name: str) -> int:
    return kcp13.COLUMN_NAMES.index(name)


def test_golden_layout_matches_contract(golden):
    assert golden.schema == kcp13.SCHEMA_ID
    assert golden.perspective == kcp13.PERSPECTIVE
    assert golden.columns == kcp13.COLUMN_NAMES
    assert golden.engine_version == kcp13.GOLDEN_ENGINE_VERSION
    assert golden.matrix().shape == (len(golden.probes), kcp13.FEATURE_COUNT)
    assert len(kcp13.COLUMN_NAMES) == kcp13.FEATURE_COUNT == 13


def test_golden_probes_are_read_from_the_committed_tsv(golden):
    rows = (kcp13.GOLDEN_DIR / "probes.tsv").read_text().splitlines()
    assert rows[0] == "id\tfen\ttags\tnote"
    tsv_by_id = {row.split("\t")[0]: row.split("\t")[1] for row in rows[1:]}
    for probe in golden.probes:
        assert probe.id in tsv_by_id
        assert probe.fen == tsv_by_id[probe.id]


def test_golden_093_probes_match_full_committed_tsv():
    golden_093 = kcp13.load_golden(kcp13.golden_path("0.9.3"))
    rows = (kcp13.GOLDEN_DIR / "probes.tsv").read_text().splitlines()
    assert [row.split("\t")[0] for row in rows[1:]] == [probe.id for probe in golden_093.probes]
    assert [row.split("\t")[1] for row in rows[1:]] == [probe.fen for probe in golden_093.probes]


def test_start_position_is_zero_except_total_material(probes):
    expected = np.zeros(kcp13.FEATURE_COUNT, dtype=np.float32)
    expected[col("total_material")] = 78.0
    np.testing.assert_array_equal(probes["start-w"].features, expected)


def test_bare_kings_are_all_zero(probes):
    np.testing.assert_array_equal(probes["kings-only-w"].features, np.zeros(13, dtype=np.float32))


@pytest.mark.parametrize(
    ("left", "right"),
    [("start-w", "start-w-6field"), ("start-w", "start-b"), ("ep-e6-w", "ep-none-w")],
)
def test_canonicalization_and_symmetry_pairs_are_identical(probes, left, right):
    np.testing.assert_array_equal(probes[left].features, probes[right].features)


def test_every_twin_equals_its_base_position(probes):
    twins = [probe for probe in probes.values() if probe.id.endswith("-twin")]
    assert len(twins) >= 7
    for twin in twins:
        base = probes[twin.id.removesuffix("-twin")]
        assert kcp13.twin_fen(base.fen) == twin.fen
        assert twin.side != base.side
        np.testing.assert_array_equal(twin.features, base.features, err_msg=twin.id)


def test_material_block_matches_the_fen_for_every_probe(golden):
    for probe in golden.probes:
        np.testing.assert_array_equal(
            probe.features[:7], kcp13.material_block(probe.fen, probe.side), err_msg=probe.id
        )


def test_side_to_move_flips_the_difference_block(probes):
    up, down = probes["knight-up-w"].features, probes["knight-down-b"].features
    assert up[col("n_diff")] == 1.0 and up[col("material_diff")] == 3.0
    np.testing.assert_array_equal(up[:6], -down[:6])
    assert up[col("total_material")] == down[col("total_material")] == 75.0
    assert up[col("mobility_diff")] == -down[col("mobility_diff")]


def test_capture_probabilities_are_probabilities(golden):
    probs = golden.matrix()[:, 9:]
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_rook_bearing_on_the_enemy_king_is_attack_for_the_mover(probes):
    f = probes["rook-king-attack-w"].features
    assert f[col("king_capture_attack")] == ONE_FACE
    assert f[col("king_capture_danger")] == 0.0
    assert f[col("queen_capture_attack")] == f[col("queen_capture_danger")] == 0.0


def test_rook_bearing_on_own_king_is_danger_for_the_mover(probes):
    f = probes["rook-king-danger-w"].features
    assert f[col("king_capture_danger")] == ONE_FACE
    assert f[col("king_capture_attack")] == 0.0


def test_piece_safety_probes_move_the_queen_columns(probes):
    attack = probes["rook-queen-attack-w"].features
    assert attack[col("queen_capture_attack")] > 0.0
    assert attack[col("queen_capture_danger")] == 0.0
    en_prise = probes["queen-en-prise-w"].features
    assert en_prise[col("queen_capture_danger")] == ONE_FACE  # the e5 pawn needs one Pawn face
    assert en_prise[col("queen_capture_attack")] == 0.0
    # Qd4-d8 then Qd8xe8 needs two Queen faces; nothing else reaches the Black king.
    assert en_prise[col("king_capture_attack")] == TWO_QUEEN_FACES


def test_load_golden_rejects_a_foreign_layout(tmp_path):
    raw = json.loads(kcp13.golden_path().read_text())
    raw["columns"] = raw["columns"][::-1]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(kcp13.ContractError, match="column layout"):
        kcp13.load_golden(path)
    raw = json.loads(kcp13.golden_path().read_text())
    raw["schema"] = "kcp-14"
    path.write_text(json.dumps(raw))
    with pytest.raises(kcp13.ContractError, match="schema"):
        kcp13.load_golden(path)


def test_twin_fen_round_trips_and_flips_castling_and_en_passant():
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w Kq e6"
    twin = kcp13.twin_fen(fen)
    assert twin == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b Qk e3"
    assert kcp13.twin_fen(twin) == fen


# --- fail-closed validators -----------------------------------------------------------------


def _reference_manifest(sha: str = "a" * 64) -> dict:
    return {
        "manifestVersion": "1.0.0",
        "modelId": "candidate",
        "modelSha256": sha,
        "featureSchema": "kcp-13",
        "featureCount": 13,
        "engineCompatibility": ">=0.9.0 <0.10.0",
        "evaluationProfile": "standard-kcp",
    }


def test_validate_manifest_accepts_the_reference_shape():
    kcp13.validate_manifest(_reference_manifest(), engine_version="0.9.2")


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"manifestVersion": "1.1.0"}, "manifestVersion"),
        ({"modelId": "  "}, "modelId"),
        ({"modelSha256": "abc"}, "modelSha256"),
        ({"featureSchema": "kcp-14"}, "featureSchema"),
        ({"featureSchema": "rawboard-768", "featureCount": 768}, "featureSchema"),
        ({"featureCount": 12}, "featureCount"),
        ({"evaluationProfile": "fast-material"}, "evaluationProfile"),
        ({"engineCompatibility": ">=0.4.0 <0.9.0"}, "does not include"),
        ({"engineCompatibility": "0.9.1"}, "does not include"),
        ({"engineCompatibility": "^0.9"}, "invalid engineCompatibility"),
        ({"engineCompatibility": ""}, "must not be blank"),
    ],
)
def test_validate_manifest_fails_closed(patch, message):
    manifest = {**_reference_manifest(), **patch}
    with pytest.raises(kcp13.ContractError, match=message):
        kcp13.validate_manifest(manifest, engine_version="0.9.2")


def test_engine_compatibility_comparators():
    assert kcp13.engine_compatible(">=0.4.0", "0.9.2")
    assert kcp13.engine_compatible("0.9.2", "0.9.2")
    assert kcp13.engine_compatible(">0.9.1 <=0.9.2", "0.9.2")
    assert not kcp13.engine_compatible(">=0.4.0 <0.5.0", "0.9.2")
    assert not kcp13.engine_compatible("<0.9.2", "0.9.2")


def _linear_model(
    path,
    width: int = 13,
    input_name: str = "input",
    output_name: str = "output",
    dynamic_batch: bool = True,
    extra_output: bool = False,
    dtype=TensorProto.FLOAT,
    opset: int = 13,
    broken_graph: bool = False,
):
    """13 -> 1 Gemm + Sigmoid with a dynamic batch axis: the smallest contract-conforming graph."""
    batch = "batch" if dynamic_batch else 1
    x = helper.make_tensor_value_info(input_name, dtype, [batch, width])
    y = helper.make_tensor_value_info(output_name, dtype, [batch, 1])
    np_dtype = np.float64 if dtype == TensorProto.DOUBLE else np.float32
    weights = numpy_helper.from_array(np.full((width, 1), 0.05, dtype=np_dtype), "W")
    bias = numpy_helper.from_array(np.zeros((1,), dtype=np_dtype), "b")
    nodes = [
        helper.make_node("Gemm", [input_name, "W", "b"], ["logit"]),
        # A broken graph references a tensor nobody produces: metadata looks fine, checker fails.
        helper.make_node("Sigmoid", ["missing" if broken_graph else "logit"], [output_name]),
    ]
    outputs = [y]
    if extra_output:
        outputs.append(helper.make_tensor_value_info("logit", dtype, [batch, 1]))
    graph = helper.make_graph(nodes, "test", [x], outputs, initializer=[weights, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_validate_onnx_contract_accepts_a_conforming_graph(tmp_path):
    kcp13.validate_onnx_contract(_linear_model(tmp_path / "ok.onnx"))
    kcp13.validate_onnx_contract(_linear_model(tmp_path / "ok18.onnx", opset=kcp13.MAX_OPSET))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"output_name": "variable"}, "exactly one output named 'output'"),
        ({"input_name": "features"}, "exactly one input named 'input'"),
        ({"width": 9}, r"shape \[batch,13\]"),
        ({"dynamic_batch": False}, "dynamic batch"),
        ({"extra_output": True}, "exactly one output"),
        ({"dtype": TensorProto.DOUBLE}, "FLOAT dtype"),
        ({"opset": 19}, "opset 19 exceeds"),
        ({"broken_graph": True}, "not a valid graph"),
    ],
)
def test_validate_onnx_contract_fails_closed(tmp_path, kwargs, message):
    path = _linear_model(tmp_path / "bad.onnx", **kwargs)
    with pytest.raises(kcp13.ContractError, match=message):
        kcp13.validate_onnx_contract(path)


def test_predict_scores_the_golden_corpus_like_the_evaluator(tmp_path, golden):
    path = _linear_model(tmp_path / "ok.onnx")
    probs = kcp13.predict(path, golden.matrix())
    assert probs.shape == (len(golden.probes),)
    assert np.isfinite(probs).all() and probs.min() >= 0.0 and probs.max() <= 1.0
    expected = 1 / (1 + np.exp(-0.05 * golden.matrix().astype(np.float64).sum(axis=1)))
    np.testing.assert_allclose(probs, expected, atol=1e-6)
    with pytest.raises(kcp13.ContractError, match=r"\[n,13\]"):
        kcp13.predict(path, golden.matrix()[:, :9])


def test_build_manifest_binds_the_digest_and_validates(tmp_path):
    path = _linear_model(tmp_path / "ok.onnx")
    manifest = kcp13.build_manifest(
        path, "candidate", ">=0.9.0 <0.10.0", provenance={"training": "test"}
    )
    kcp13.validate_manifest(manifest, "0.9.2")
    kcp13.verify_model_digest(path, manifest)
    assert manifest["modelSha256"] == kcp13.sha256_of(path)
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(kcp13.ContractError, match="SHA-256 mismatch"):
        kcp13.verify_model_digest(path, manifest)


def _golden_files() -> list[Path]:
    return sorted(kcp13.GOLDEN_DIR.glob("golden-engine-*.json"))


def _engine_of(path: Path) -> str:
    return path.name.removeprefix("golden-engine-").removesuffix(".json")


def _release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


@pytest.mark.parametrize("path", _golden_files(), ids=_engine_of)
def test_every_golden_declares_the_engine_named_by_its_file(path):
    corpus = kcp13.load_golden(path)
    engine = _engine_of(path)
    assert corpus.engine_version == engine
    assert corpus.engine_artifact == f"com.fortemate:dicechess-engine_3:{engine}"


def test_goldens_agree_bit_for_bit_on_every_shared_probe():
    """Every engine release replays the corpus exactly; a drifting extractor fails here first."""
    corpora = {_engine_of(path): kcp13.load_golden(path) for path in _golden_files()}
    assert {"0.9.2", "0.9.3", "0.12.0"} <= corpora.keys()
    newest = max(corpora, key=_release)
    reference = corpora[newest].by_id()
    for engine, corpus in corpora.items():
        if engine == newest:
            continue
        probes = corpus.by_id()
        shared = sorted(probes.keys() & reference.keys())
        assert shared, f"golden {engine} shares no probe with {newest}"
        for probe_id in shared:
            probe, expected = probes[probe_id], reference[probe_id]
            assert probe.fen == expected.fen and probe.side == expected.side
            assert probe.features.tobytes() == expected.features.tobytes(), (
                f"engine {engine} differs from {newest} on probe {probe_id}"
            )


def test_the_newest_golden_covers_every_committed_probe():
    newest = max((_engine_of(path) for path in _golden_files()), key=_release)
    corpus = kcp13.load_golden(kcp13.golden_path(newest))
    rows = (kcp13.GOLDEN_DIR / "probes.tsv").read_text().splitlines()[1:]
    assert [row.split("\t")[0] for row in rows] == [probe.id for probe in corpus.probes]


# --- Manifest 1.1.0 (#44) ---------------------------------------------------------------------

MANIFESTS = Path(kcp13.__file__).resolve().parents[3] / "tests/fixtures/manifests"
ENGINE_VERSION = "0.12.0"


def _engine_fixture(name):
    return json.loads((MANIFESTS / f"{name}.json").read_text(encoding="utf-8"))


def _serviceable(manifest):
    """The engine omits `evaluationProfile`; the service this contract mirrors requires it."""
    return {**manifest, "evaluationProfile": kcp13.PROFILE_ID}


def test_a_legacy_manifest_still_validates_unchanged():
    kcp13.validate_manifest(
        _engine_fixture("synthetic_kcp13_value_legacy_manifest"), ENGINE_VERSION
    )


@pytest.mark.parametrize(
    "name",
    [
        "synthetic_kcp13_value_manifest",
        "synthetic_kcp13_collapse_manifest",
        "synthetic_kcp13_renamed_tensors_manifest",
    ],
)
def test_the_engines_committed_manifests_validate(name):
    """Every 1.1.0 field the two readers share is agreed; only the service-side one is added."""
    kcp13.validate_manifest(_serviceable(_engine_fixture(name)), ENGINE_VERSION)


@pytest.mark.parametrize(
    "name",
    [
        "synthetic_kcp13_value_manifest",
        "synthetic_kcp13_collapse_manifest",
        "synthetic_kcp13_renamed_tensors_manifest",
    ],
)
def test_the_engines_manifests_are_refused_only_for_the_field_the_engine_does_not_read(name):
    """Documents the one divergence, so it cannot drift into an unnoticed difference."""
    manifest = _engine_fixture(name)
    with pytest.raises(kcp13.ContractError, match="evaluationProfile"):
        kcp13.validate_manifest(manifest, ENGINE_VERSION)


@pytest.mark.parametrize("field", ["modelRole", "perspective"])
def test_a_current_manifest_missing_a_required_field_names_it(field):
    manifest = _serviceable(_engine_fixture("synthetic_kcp13_value_manifest"))
    del manifest[field]
    with pytest.raises(kcp13.ContractError, match=f"requires field '{field}'"):
        kcp13.validate_manifest(manifest, ENGINE_VERSION)


@pytest.mark.parametrize("field", ["modelRole", "perspective"])
def test_a_legacy_manifest_carrying_a_current_field_is_refused_not_defaulted(field):
    """A 1.0.0-only reader ignores the field, so honouring it makes one file mean two things."""
    manifest = _engine_fixture("synthetic_kcp13_value_legacy_manifest")
    manifest[field] = {"modelRole": "chance-collapse", "perspective": "side-to-move"}[field]
    with pytest.raises(kcp13.ContractError, match=f"does not define field '{field}'"):
        kcp13.validate_manifest(manifest, ENGINE_VERSION)


def test_an_unknown_role_fails_closed_rather_than_defaulting():
    manifest = _serviceable(_engine_fixture("synthetic_kcp13_value_manifest"))
    manifest["modelRole"] = "leaf-evaluator"
    with pytest.raises(kcp13.ContractError, match="unknown modelRole"):
        kcp13.validate_manifest(manifest, ENGINE_VERSION)


def test_an_unsupported_perspective_fails_closed():
    manifest = _serviceable(_engine_fixture("synthetic_kcp13_value_manifest"))
    manifest["perspective"] = "white"
    with pytest.raises(kcp13.ContractError, match="unsupported perspective"):
        kcp13.validate_manifest(manifest, ENGINE_VERSION)


def test_a_version_outside_the_supported_set_is_refused_before_any_other_field():
    manifest = {"manifestVersion": "2.0.0"}
    with pytest.raises(kcp13.ContractError, match="unsupported manifestVersion"):
        kcp13.validate_manifest(manifest, ENGINE_VERSION)


@pytest.mark.parametrize("field", ["inputName", "outputName"])
def test_a_blank_tensor_name_is_refused(field):
    manifest = _serviceable(_engine_fixture("synthetic_kcp13_value_manifest"))
    manifest[field] = "  "
    with pytest.raises(kcp13.ContractError, match=f"{field} must not be blank"):
        kcp13.validate_manifest(manifest, ENGINE_VERSION)


def test_tensor_names_default_to_the_contracts_own():
    legacy = _engine_fixture("synthetic_kcp13_value_legacy_manifest")
    assert kcp13.manifest_tensor_names(legacy) == (kcp13.INPUT_NAME, kcp13.OUTPUT_NAME)
    renamed = _engine_fixture("synthetic_kcp13_renamed_tensors_manifest")
    assert kcp13.manifest_tensor_names(renamed) == (renamed["inputName"], renamed["outputName"])


def test_the_role_decides_the_version_a_package_declares(tmp_path):
    """A position model keeps writing 1.0.0: the deployed service can parse nothing else."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a graph; only the digest is read here")
    position = kcp13.build_manifest(model, "m", ">=0.12.0 <1.0.0")
    assert position["manifestVersion"] == kcp13.LEGACY_MANIFEST_VERSION
    assert "modelRole" not in position

    collapse = kcp13.build_manifest(
        model, "m", ">=0.12.0 <1.0.0", model_role=kcp13.ROLE_CHANCE_COLLAPSE
    )
    assert collapse["manifestVersion"] == kcp13.CURRENT_MANIFEST_VERSION
    assert collapse["modelRole"] == kcp13.ROLE_CHANCE_COLLAPSE
    assert collapse["perspective"] == kcp13.PERSPECTIVE
    assert (collapse["inputName"], collapse["outputName"]) == (kcp13.INPUT_NAME, kcp13.OUTPUT_NAME)

    renamed = kcp13.build_manifest(model, "m", ">=0.12.0 <1.0.0", input_name="features")
    assert renamed["manifestVersion"] == kcp13.CURRENT_MANIFEST_VERSION


def test_building_an_unknown_role_is_refused(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"digest only")
    role = "leaf-evaluator"
    with pytest.raises(kcp13.ContractError, match="unknown modelRole"):
        kcp13.build_manifest(model, "m", ">=0.12.0 <1.0.0", model_role=role)


def test_a_built_manifest_validates_for_every_role(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"digest only")
    for role in kcp13.SUPPORTED_ROLES:
        manifest = kcp13.build_manifest(model, "m", ">=0.12.0 <1.0.0", model_role=role)
        kcp13.validate_manifest(manifest, ENGINE_VERSION)
