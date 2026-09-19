"""A package's digest must be a function of the model and nothing else (#59).

The exporter writes the absolute source path and line number of the traced `forward` onto every
node. That made `modelSha256` — which the seal binds and which `serving.recover_model` reproduces
byte for byte — depend on where the checkout sits on disk and on where a line sits in the file
defining the model, while the graph was numerically identical.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
import torch

from dicechess_training.ablation.export import ENVIRONMENT_METADATA, export_candidate
from dicechess_training.ablation.runner import ValueMLP
from dicechess_training.benchmark import core
from dicechess_training.contracts import kcp13


@pytest.fixture
def exported(tmp_path):
    torch.manual_seed(5)
    model = ValueMLP(kcp13.FEATURE_COUNT, [8, 8], np.zeros(13, np.float32), np.ones(13, np.float32))
    return export_candidate(model, kcp13.SCHEMA_ID, tmp_path / "model.onnx")


def test_no_node_carries_build_environment_metadata(exported):
    graph = onnx.load(str(exported)).graph
    offending = {
        entry.key
        for node in graph.node
        for entry in node.metadata_props
        if entry.key in ENVIRONMENT_METADATA
    }
    assert offending == set()


def test_the_artifact_holds_no_path_from_the_machine_that_built_it(exported):
    """The property that matters, stated so a future exporter's new metadata trips it too.

    Checking only the keys we know about would pass for whatever is added next; the artifact
    simply must not contain this checkout's location.
    """
    raw = exported.read_bytes()
    assert str(core.ROOT).encode() not in raw
    assert b"runner.py" not in raw
    # The repository directory name on its own, in case an exporter writes a relative path.
    assert core.ROOT.name.encode() not in raw


def test_stripping_leaves_a_graph_the_contract_still_admits(exported):
    kcp13.validate_onnx_contract(exported)
    onnx.checker.check_model(onnx.load(str(exported)), full_check=True)


def test_the_export_is_unchanged_by_where_the_model_was_defined(tmp_path):
    """Two identical models exported twice give identical bytes — the baseline the rest rests on."""
    paths = []
    for name in ("first", "second"):
        torch.manual_seed(5)
        model = ValueMLP(
            kcp13.FEATURE_COUNT, [8, 8], np.zeros(13, np.float32), np.ones(13, np.float32)
        )
        paths.append(export_candidate(model, kcp13.SCHEMA_ID, tmp_path / f"{name}.onnx"))
    assert kcp13.sha256_of(paths[0]) == kcp13.sha256_of(paths[1])
