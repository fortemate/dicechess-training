"""Export an ablation candidate to the serving graph of ADR 0001.

The exported graph is what makes standardisation safe to adopt (#27): the training statistics
travel inside the model as a constant affine transform, so the consumer keeps sending **raw**
schema features and the `input [batch, N]` -> `output [batch, 1]` contract is untouched. A
pipeline that standardised in Python and exported the bare network instead would serve silently
wrong probabilities, which is why the parity check below runs on the engine's golden probes
rather than on random vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dicechess_training.ablation.runner import ValueMLP, predict
from dicechess_training.contracts import SCHEMA_CONTRACTS

MAX_OPSET = 18


def export_candidate(model: ValueMLP, schema_id: str, path: str | Path) -> Path:
    """Write `model` to ONNX under the contract names for `schema_id`."""
    contract = SCHEMA_CONTRACTS[schema_id]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros((1, contract.FEATURE_COUNT), dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        str(destination),
        input_names=[contract.INPUT_NAME],
        output_names=[contract.OUTPUT_NAME],
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        opset_version=MAX_OPSET,
        dynamo=True,
        # Weights stay inside the file. The exporter defaults to a sidecar `.onnx.data`, which
        # the serving contract cannot carry: the evaluator mounts one artifact and binds one
        # SHA-256 to it, so externalised weights would travel unchecked or not at all.
        external_data=False,
    )
    _strip_build_environment(destination)
    return destination


def golden_probe_features(schema_id: str, engine_version: str | None = None) -> np.ndarray:
    """Authored golden probes for `schema_id` as a raw feature matrix."""
    contract = SCHEMA_CONTRACTS[schema_id]
    corpus = (
        contract.load_golden(contract.golden_path(engine_version))
        if engine_version
        else contract.load_golden()
    )
    authored = [probe for probe in corpus.probes if not probe.id.startswith("sample-")]
    return np.stack([probe.features for probe in authored])


def probe_parity(
    model: ValueMLP,
    schema_id: str,
    path: str | Path,
    engine_version: str | None = None,
) -> float:
    """Largest absolute probability difference between the model and its exported graph."""
    contract = SCHEMA_CONTRACTS[schema_id]
    features = golden_probe_features(schema_id, engine_version)
    torch_probs = predict(model, features)
    onnx_probs = contract.predict(path, features)
    return float(np.abs(torch_probs - onnx_probs).max())


#: Metadata the exporter attaches that describes the machine rather than the model. Named
#: explicitly: stripping everything unfamiliar would silently discard whatever a future exporter
#: adds that a consumer does need.
ENVIRONMENT_METADATA = ("pkg.torch.onnx.stack_trace",)


def _strip_build_environment(path: Path) -> None:
    """Remove metadata that makes the bytes a function of where the build happened.

    `torch.onnx.export(dynamo=True)` writes a `pkg.torch.onnx.stack_trace` entry onto every node,
    holding the absolute source path and the line number of the traced `forward`. That makes
    `modelSha256` depend on the checkout's location on disk and on where a line sits in the file
    defining the model — measured: an unused import above `forward` changes the digest, and so does
    building the same commit from a second checkout, while the graphs agree to 0.0 (#59).

    Two things follow, and both matter. A digest that moves for reasons unrelated to the model
    cannot bind it, and `serving.recover_model` compares exactly those bytes. And the artifact
    would otherwise carry an absolute path from the build machine, which the publication boundary
    treats as private wherever it is written.
    """
    import onnx

    model = onnx.load(str(path))
    for node in model.graph.node:
        keep = [entry for entry in node.metadata_props if entry.key not in ENVIRONMENT_METADATA]
        if len(keep) != len(node.metadata_props):
            del node.metadata_props[:]
            node.metadata_props.extend(keep)
    onnx.save(model, str(path))
