"""What a pre-ranking artifact must be, stated as refusals.

The evaluator's contract is the wrong template for this model in four places — role, manifest
version, output meaning and feature schema — so each of those has a test that fails if someone
copies it across.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from dicechess_training.contracts import SCHEMA_CONTRACTS, kcp13, prerank

ENGINE = prerank.DEFAULT_ENGINE_VERSION


def _ranker(
    path: Path,
    width: int | None = None,
    input_name: str = prerank.INPUT_NAME,
    output_name: str = prerank.OUTPUT_NAME,
    dynamic_batch: bool = True,
    dtype=TensorProto.FLOAT,
    opset: int = 13,
    scale: float = 1.0,
) -> Path:
    """The smallest conforming ranker: width -> 1 Gemm, and no sigmoid.

    The missing sigmoid is the point — this graph emits a score, and the contract must accept one
    that leaves the unit interval.
    """
    width = prerank.feature_count(ENGINE) if width is None else width
    batch = "batch" if dynamic_batch else 1
    x = helper.make_tensor_value_info(input_name, dtype, [batch, width])
    y = helper.make_tensor_value_info(output_name, dtype, [batch, 1])
    np_dtype = np.float64 if dtype == TensorProto.DOUBLE else np.float32
    weights = numpy_helper.from_array(np.full((width, 1), scale, dtype=np_dtype), "W")
    bias = numpy_helper.from_array(np.zeros((1,), dtype=np_dtype), "b")
    nodes = [helper.make_node("Gemm", [input_name, "W", "b"], [output_name])]
    graph = helper.make_graph(nodes, "ranker", [x], [y], initializer=[weights, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _manifest(path: Path, **overrides) -> dict:
    manifest = prerank.build_manifest(
        path, "prerank-test", ">=0.12.0 <0.13.0", {"seed": "1"}, ENGINE
    )
    manifest.update(overrides)
    return manifest


def test_columns_come_from_the_committed_corpus() -> None:
    assert prerank.columns(ENGINE) == tuple(
        json.loads(prerank.golden_path(ENGINE).read_text(encoding="utf-8"))["columns"]
    )
    assert prerank.feature_count(ENGINE) == 9


def test_the_student_sees_the_evaluators_cheap_columns() -> None:
    """rich-9 is the kcp-13 prefix; the four columns it drops are the 216-outcome ones."""
    assert prerank.columns(ENGINE) == kcp13.COLUMN_NAMES[: prerank.feature_count(ENGINE)]


def test_an_unknown_engine_has_no_layout_to_offer() -> None:
    with pytest.raises(kcp13.ContractError, match="no committed golden corpus"):
        prerank.columns("9.9.9")


def test_the_contract_is_not_a_value_model_schema() -> None:
    """Registering it would say an enriched shard can be read with it, which is false."""
    assert prerank.SCHEMA_ID not in SCHEMA_CONTRACTS


def test_a_built_manifest_validates(tmp_path: Path) -> None:
    model = _ranker(tmp_path / "m.onnx")
    manifest = _manifest(model)
    prerank.validate_manifest(manifest, ENGINE)
    assert manifest["manifestVersion"] == "1.1.0"
    assert manifest["modelRole"] == "move-prerank"
    assert manifest["perspective"] == "side-to-move"
    assert (manifest["inputName"], manifest["outputName"]) == prerank.manifest_tensor_names(
        manifest
    )
    assert "calibration" not in manifest
    assert "evaluationProfile" not in manifest


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"manifestVersion": "1.0.0"}, "a pre-ranker declares a role"),
        ({"modelRole": "position-value"}, "cannot serve as 'move-prerank'"),
        ({"modelRole": "chance-collapse"}, "cannot serve as 'move-prerank'"),
        ({"perspective": "white"}, "unsupported perspective"),
        ({"modelId": "  "}, "modelId must not be blank"),
        ({"modelSha256": "abc"}, "64 hexadecimal"),
        # 64 characters and not one of them hexadecimal: the length check passes and the
        # digest is still unusable, which is where a parse would raise the wrong exception.
        ({"modelSha256": "z" * 64}, "64 hexadecimal"),
        ({"modelSha256": 12345}, "64 hexadecimal"),
        ({"featureSchema": "kcp-13"}, "unsupported featureSchema"),
        ({"featureCount": 13}, "unsupported featureCount"),
        ({"calibration": {"temperature": 1.12}}, "nothing to calibrate"),
        ({"engineCompatibility": ">=0.4.0 <0.5.0"}, "does not include engine version"),
    ],
)
def test_a_manifest_that_misdescribes_the_artifact_is_refused(
    tmp_path: Path, override: dict, message: str
) -> None:
    model = _ranker(tmp_path / "m.onnx")
    with pytest.raises(kcp13.ContractError) as error:
        prerank.validate_manifest(_manifest(model, **override), ENGINE)
    assert message in str(error.value)


def test_a_conforming_graph_is_accepted(tmp_path: Path) -> None:
    prerank.validate_onnx_contract(_ranker(tmp_path / "ok.onnx"))
    prerank.validate_onnx_contract(_ranker(tmp_path / "ok18.onnx", opset=prerank.MAX_OPSET))


def test_declared_tensor_names_are_honoured(tmp_path: Path) -> None:
    model = _ranker(tmp_path / "named.onnx", input_name="rows", output_name="score")
    manifest = _manifest(model, inputName="rows", outputName="score")
    names = prerank.manifest_tensor_names(manifest)
    prerank.validate_onnx_contract(model, *names)
    with pytest.raises(kcp13.ContractError, match="exactly one input named"):
        prerank.validate_onnx_contract(model)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 13}, "shape [batch,9]"),
        ({"dynamic_batch": False}, "dynamic batch"),
        ({"dtype": TensorProto.DOUBLE}, "FLOAT dtype"),
        ({"opset": prerank.MAX_OPSET + 1}, "exceeds the maximum"),
    ],
)
def test_a_graph_the_engine_could_not_open_is_refused(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    model = _ranker(tmp_path / "bad.onnx", **kwargs)
    with pytest.raises(kcp13.ContractError) as error:
        prerank.validate_onnx_contract(model)
    assert message in str(error.value)


def test_scores_are_not_clamped(tmp_path: Path) -> None:
    """The difference from the evaluator's contract that matters most.

    A ranker's output means nothing on its own and everything in comparison, so values outside
    [0, 1] are ordinary. Clipping them — as a probability contract must — would collapse the
    ordering this model exists to produce.
    """
    model = _ranker(tmp_path / "wide.onnx", scale=10.0)
    features = np.zeros((3, prerank.feature_count(ENGINE)), dtype=np.float32)
    features[0, 0] = 5.0  # 50.0
    features[1, 0] = -5.0  # -50.0
    scores = prerank.score(model, features)
    assert scores[0] > 1.0 and scores[1] < 0.0
    assert scores[0] > scores[2] > scores[1]


def test_scores_refuse_the_wrong_width(tmp_path: Path) -> None:
    model = _ranker(tmp_path / "m.onnx")
    with pytest.raises(kcp13.ContractError, match=r"features must be \[n,9\]"):
        prerank.score(model, np.zeros((2, 13), dtype=np.float32))


def test_order_survives_a_monotone_rescale(tmp_path: Path) -> None:
    """Why calibration is meaningless here: scaling every score changes the numbers and not the
    answer, so there is no scale to be right about."""
    small = _ranker(tmp_path / "small.onnx", scale=0.5)
    large = _ranker(tmp_path / "large.onnx", scale=50.0)
    features = np.arange(12, dtype=np.float32).reshape(-1, 4)
    features = np.pad(features, ((0, 0), (0, prerank.feature_count(ENGINE) - 4)))
    assert list(np.argsort(prerank.score(small, features))) == list(
        np.argsort(prerank.score(large, features))
    )
