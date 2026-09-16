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
    )
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
