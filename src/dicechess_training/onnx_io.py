"""ONNX export and PyTorch <-> onnxruntime parity checking.

ONNX is the bridge to production: the Scala engine consumes these files via
ONNX Runtime on JVM, Workers, and Raspberry Pi. The exported graph includes
the sigmoid, so consumers read a mover win probability directly. The parity
check exists because a model that silently disagrees with its exported self is
the most expensive bug this pipeline can ship.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from .features import FEATURE_DIM
from .model import ValueMLP

INPUT_NAME = "features"
OUTPUT_NAME = "win_probability"


def export_value_model(model: ValueMLP, path: str) -> None:
    """Export the model (with sigmoid) to ONNX with a dynamic batch axis."""
    wrapped = nn.Sequential(model.net, nn.Sigmoid())
    wrapped.eval()
    example = torch.zeros((1, FEATURE_DIM), dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        (example,),
        path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        opset_version=18,
        dynamo=True,
    )


def onnx_parity(model: ValueMLP, path: str, n: int = 256, seed: int = 0) -> float:
    """Max |torch - onnxruntime| win-probability difference on random inputs."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, FEATURE_DIM), dtype=np.float32)
    with torch.no_grad():
        torch_probs = model.predict_proba(torch.from_numpy(x)).numpy().reshape(-1)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onnx_probs = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0].reshape(-1)
    return float(np.abs(torch_probs - onnx_probs).max())
