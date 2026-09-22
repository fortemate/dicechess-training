"""Turn trained weights into the artifact the engine's seam can open.

Three things happen here that are not "save the model", and each of them is a lesson this
repository paid for once already.

**The graph is float32, the training was not.** The contract's tensors are FLOAT because that is
what the serving runtime binds, so the export converts first and then *re-measures*: a ranking is
decided by comparisons, and a comparison between two nearly equal scores can flip when the
arithmetic narrows. The numbers that belong in a report are the artifact's, not the checkpoint's.

**The bytes must be a function of the model, not of the machine.** `torch.onnx.export(dynamo=True)`
writes the absolute source path and line number of the traced `forward` onto every node, which made
`modelSha256` depend on where the checkout happened to live and left packages built before #60
permanently unrecoverable. That metadata is stripped, exactly as `ablation.export` strips it.

**The batch axis stays dynamic.** The seam feeds every legal turn through this model in bounded
chunks — up to 2,420 rows from one root — so a graph pinned to a row count would refuse the only
workload it has.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from dicechess_training.contracts import prerank as contract
from dicechess_training.prerank.model import PreRankMLP

MODEL_FILE = "model.onnx"
MANIFEST_FILE = "manifest.json"

#: Metadata the exporter attaches that describes the machine rather than the model. Named
#: explicitly rather than filtered by prefix: stripping everything unfamiliar would silently
#: discard whatever a future exporter adds that a consumer does need.
ENVIRONMENT_METADATA = ("pkg.torch.onnx.stack_trace",)


def load_ranker(weights: str | Path) -> PreRankMLP:
    """Rebuild a trained ranker from its checkpoint alone.

    The shapes carry the architecture, so nothing beside the file has to remember it: the width
    of each `Linear` is the hidden size, and the presence of the standardisation buffers says
    whether the model carries its own scaler. A checkpoint that needed a separate note saying how
    to read it would be a checkpoint that outlives the note.
    """
    state = torch.load(Path(weights), map_location="cpu")
    linears = sorted(
        (key for key in state if key.startswith("net.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if not linears:
        raise ValueError("the checkpoint holds no layers")
    hidden = [int(state[key].shape[0]) for key in linears[:-1]]
    model = PreRankMLP(
        input_dim=int(state[linears[0]].shape[1]),
        hidden_dims=hidden,
        feature_mean=state["feature_mean"].numpy() if "feature_mean" in state else None,
        feature_scale=state["feature_scale"].numpy() if "feature_scale" in state else None,
    ).double()
    model.load_state_dict(state)
    model.eval()
    return model


def as_served(model: PreRankMLP) -> PreRankMLP:
    """A float32 copy, which is what the contract's tensors are and what the runtime will bind."""
    served = copy.deepcopy(model).float()
    served.eval()
    return served


def export_ranker(
    model: PreRankMLP,
    path: str | Path,
    engine_version: str = contract.DEFAULT_ENGINE_VERSION,
) -> Path:
    """Write the graph under the contract's names, then hold it to the contract."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    served = as_served(model)
    example = torch.zeros((1, contract.feature_count(engine_version)), dtype=torch.float32)
    torch.onnx.export(
        served,
        (example,),
        str(destination),
        input_names=[contract.INPUT_NAME],
        output_names=[contract.OUTPUT_NAME],
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        opset_version=contract.MAX_OPSET,
        dynamo=True,
        # Weights stay inside the file: the seam mounts one artifact and binds one digest to it,
        # so externalised weights would travel unchecked or not at all.
        external_data=False,
    )
    _strip_build_environment(destination)
    contract.validate_onnx_contract(destination, engine_version=engine_version)
    return destination


def _strip_build_environment(path: Path) -> None:
    """Remove the metadata that makes the bytes a function of where the build happened."""
    import onnx

    model = onnx.load(str(path))
    changed = False
    for node in model.graph.node:
        keep = [entry for entry in node.metadata_props if entry.key not in ENVIRONMENT_METADATA]
        if len(keep) != len(node.metadata_props):
            del node.metadata_props[:]
            node.metadata_props.extend(keep)
            changed = True
    if changed:
        onnx.save(model, str(path))


def probe_parity(
    model: PreRankMLP,
    path: str | Path,
    engine_version: str = contract.DEFAULT_ENGINE_VERSION,
) -> float:
    """Largest absolute difference between the model and its graph, on the engine's own probes.

    The golden corpus rather than random vectors, for the reason ADR 0001 gives: the probes are
    positions somebody chose to be structurally different, and a disagreement on them is a
    disagreement about something.
    """
    corpus = json.loads(contract.golden_path(engine_version).read_text(encoding="utf-8"))
    features = np.stack(
        [probe["features"] for probe in corpus["probes"] if not probe["id"].startswith("sample-")]
    ).astype(np.float32)
    with torch.no_grad():
        theirs = as_served(model)(torch.from_numpy(features)).reshape(-1).numpy()
    ours = contract.score(path, features, engine_version=engine_version)
    return float(np.abs(theirs.astype(np.float64) - ours).max())


def write_package(
    model: PreRankMLP,
    directory: str | Path,
    model_id: str,
    engine_compatibility: str,
    provenance: dict[str, str] | None = None,
    engine_version: str = contract.DEFAULT_ENGINE_VERSION,
) -> dict:
    """Export the graph and the manifest that describes it, and refuse to leave a bad pair.

    The manifest is validated after it is written, against the graph it names, because a manifest
    nobody checked is a claim rather than a description — and `modelSha256` binding the wrong
    bytes is the failure that no consumer can diagnose.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = export_ranker(model, directory / MODEL_FILE, engine_version)
    manifest = contract.build_manifest(
        model_path=model_path,
        model_id=model_id,
        engine_compatibility=engine_compatibility,
        provenance=provenance,
        engine_version=engine_version,
    )
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract.validate_manifest(manifest, engine_version=engine_version)
    if contract.sha256_of(model_path) != manifest["modelSha256"]:
        raise contract.ContractError("the manifest does not describe the graph beside it")
    return manifest
