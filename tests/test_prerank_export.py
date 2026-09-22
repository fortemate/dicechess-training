"""What the exported artifact has to be, held to the contract the engine will open it with.

The expensive failures here are all silent ones: a graph whose batch axis is pinned, a digest that
depends on the machine that built it, a manifest that describes different bytes than the ones
beside it. None of them looks wrong until something far away refuses to load, so each gets a test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dicechess_training.contracts import prerank as contract
from dicechess_training.prerank.export import (
    ENVIRONMENT_METADATA,
    MANIFEST_FILE,
    MODEL_FILE,
    as_served,
    export_ranker,
    load_ranker,
    probe_parity,
    write_package,
)
from dicechess_training.prerank.model import PreRankMLP

ENGINE = "0.12.0"


def _model(seed: int = 0, standardised: bool = True) -> PreRankMLP:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    width = contract.feature_count(ENGINE)
    statistics = (
        {
            "feature_mean": rng.normal(size=width).astype(np.float32),
            "feature_scale": np.abs(rng.normal(size=width)).astype(np.float32) + 0.5,
        }
        if standardised
        else {}
    )
    return PreRankMLP(width, [8, 8], **statistics).double()


def test_the_exported_graph_satisfies_the_contract(tmp_path) -> None:
    path = export_ranker(_model(), tmp_path / MODEL_FILE, ENGINE)
    contract.validate_onnx_contract(path, engine_version=ENGINE)


# The seam feeds every legal turn through this model in bounded chunks — up to 2,420 rows from one
# root — so a graph pinned to a row count would refuse the only workload it has.
@pytest.mark.parametrize("rows", [1, 7, 512])
def test_the_batch_axis_is_dynamic(tmp_path, rows: int) -> None:
    path = export_ranker(_model(), tmp_path / MODEL_FILE, ENGINE)
    features = np.zeros((rows, contract.feature_count(ENGINE)), dtype=np.float32)
    assert contract.score(path, features, engine_version=ENGINE).shape == (rows,)


def _probe_features() -> np.ndarray:
    corpus = json.loads(contract.golden_path(ENGINE).read_text(encoding="utf-8"))
    return np.stack([probe["features"] for probe in corpus["probes"]]).astype(np.float32)


def test_the_graph_agrees_with_the_model_it_came_from(tmp_path) -> None:
    """Relative to the spread of the scores, not to an absolute number.

    A ranker's output is unbounded by design and its units mean nothing on their own — only
    differences within a group do — so "agrees to 1e-5" is a claim about nothing until it is
    divided by the range those scores span. Both sides are float32 here, so the honest bound is
    a few ulps of that range.
    """
    model = _model()
    path = export_ranker(model, tmp_path / MODEL_FILE, ENGINE)
    features = _probe_features()
    spread = float(np.ptp(contract.score(path, features, engine_version=ENGINE)))
    assert probe_parity(model, path, ENGINE) < 1e-6 * max(spread, 1.0)


def test_the_graph_does_not_invert_a_pair_the_model_separated(tmp_path) -> None:
    """The property that actually matters, stated so that it is true.

    A ranker is its ordering, so the graph must not disagree with the model about which of two
    candidates is better. It may disagree about two candidates the model could not separate —
    scores within the resolution of float32 arithmetic can come out either way, and an ordering
    between them is not a decision anybody made.

    An earlier version of this test demanded the *whole* permutation match, and it was flaky
    rather than wrong: it passed three CI runs and failed the fourth on the same platform,
    because ONNX Runtime and torch can accumulate the same sum in different orders. Asserting no
    inversion of a *separated* pair is the same guarantee without the coin flip.
    """
    model = _model(9)
    path = export_ranker(model, tmp_path / MODEL_FILE, ENGINE)
    features = _probe_features()
    with torch.no_grad():
        theirs = as_served(model)(torch.from_numpy(features)).reshape(-1).numpy().astype(np.float64)
    ours = contract.score(path, features, engine_version=ENGINE)

    separated = 1e-5 * max(float(np.ptp(theirs)), 1.0)
    inversions = [
        (i, j)
        for i in range(len(theirs))
        for j in range(i + 1, len(theirs))
        if abs(theirs[i] - theirs[j]) > separated and (theirs[i] > theirs[j]) != (ours[i] > ours[j])
    ]
    assert not inversions, f"the graph reordered separated probes: {inversions}"
    # And the tie band has to be narrow enough that the test is asserting something: if every
    # pair were "unseparated" this would pass vacuously.
    pairs = len(theirs) * (len(theirs) - 1) // 2
    checked = sum(
        1
        for i in range(len(theirs))
        for j in range(i + 1, len(theirs))
        if abs(theirs[i] - theirs[j]) > separated
    )
    assert checked > 0.9 * pairs, f"only {checked} of {pairs} pairs were separated enough to check"


def test_the_output_is_not_squashed_into_an_interval(tmp_path) -> None:
    """An ordering is invariant to monotone transforms, so there is nothing to calibrate; the
    contract refuses a calibration block on this role for the same reason."""
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(60.0)
    path = export_ranker(model, tmp_path / MODEL_FILE, ENGINE)
    features = np.random.default_rng(1).normal(size=(64, contract.feature_count(ENGINE)))
    scores = contract.score(path, features.astype(np.float32), engine_version=ENGINE)
    assert scores.max() > 1.0 or scores.min() < 0.0


# `torch.onnx.export(dynamo=True)` writes the absolute source path and line number of the traced
# `forward` onto every node. That made `modelSha256` a function of where the checkout lived and
# left every package built before #60 permanently unrecoverable.
def test_no_node_carries_the_machine_it_was_built_on(tmp_path) -> None:
    import onnx

    path = export_ranker(_model(), tmp_path / MODEL_FILE, ENGINE)
    graph = onnx.load(str(path))
    for node in graph.graph.node:
        assert not [entry for entry in node.metadata_props if entry.key in ENVIRONMENT_METADATA]
    assert str(tmp_path) not in path.read_bytes().decode("latin-1")


def test_the_same_model_exports_to_the_same_bytes(tmp_path) -> None:
    """A digest binds the model only if the bytes are a function of the model."""
    first = export_ranker(_model(3), tmp_path / "a" / MODEL_FILE, ENGINE)
    second = export_ranker(_model(3), tmp_path / "b" / MODEL_FILE, ENGINE)
    assert first.read_bytes() == second.read_bytes()


def test_two_different_models_do_not(tmp_path) -> None:
    first = export_ranker(_model(3), tmp_path / "a" / MODEL_FILE, ENGINE)
    second = export_ranker(_model(4), tmp_path / "b" / MODEL_FILE, ENGINE)
    assert first.read_bytes() != second.read_bytes()


def test_a_package_describes_the_graph_beside_it(tmp_path) -> None:
    manifest = write_package(
        _model(),
        tmp_path,
        model_id="prerank-test",
        engine_compatibility=">=0.12.0",
        provenance={"protocol": "playground-prerank-v1"},
        engine_version=ENGINE,
    )
    written = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert written == manifest
    assert manifest["modelSha256"] == contract.sha256_of(tmp_path / MODEL_FILE)
    contract.validate_manifest(manifest, engine_version=ENGINE)


def test_the_package_declares_the_role_the_seam_wants(tmp_path) -> None:
    """Ordering candidates and valuing positions are different jobs; the engine refuses a package
    whose declared role is not the one the seam asked for."""
    manifest = write_package(
        _model(), tmp_path, model_id="prerank-test", engine_compatibility=">=0.12.0"
    )
    assert manifest["modelRole"] == "move-prerank"
    assert manifest["manifestVersion"] == "1.1.0"
    assert manifest["featureSchema"] == "rich-9-v1"
    assert manifest["featureCount"] == 9
    assert "calibration" not in manifest


def test_the_served_copy_is_float32_and_leaves_the_original_alone(tmp_path) -> None:
    """The contract's tensors are FLOAT, and a training checkpoint this deep-copies from should
    still be usable afterwards."""
    model = _model()
    served = as_served(model)
    assert next(served.parameters()).dtype == torch.float32
    assert next(model.parameters()).dtype == torch.float64
    assert not served.training


def test_a_checkpoint_rebuilds_without_being_told_its_shape(tmp_path) -> None:
    model = _model(5)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    rebuilt = load_ranker(tmp_path / "weights.pt")
    features = torch.from_numpy(
        np.random.default_rng(2).normal(size=(16, contract.feature_count(ENGINE)))
    )
    with torch.no_grad():
        assert torch.allclose(model(features), rebuilt(features))


def test_a_checkpoint_without_a_scaler_rebuilds_too(tmp_path) -> None:
    model = _model(6, standardised=False)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    rebuilt = load_ranker(tmp_path / "weights.pt")
    assert rebuilt.standardised is False


def test_an_empty_checkpoint_is_refused(tmp_path) -> None:
    torch.save({}, tmp_path / "weights.pt")
    with pytest.raises(ValueError, match="no layers"):
        load_ranker(tmp_path / "weights.pt")
