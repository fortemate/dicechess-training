"""Training loop and holdout metrics for the value net.

Loss is binary cross-entropy with SOFT targets: outcomes are 0 / 0.5 / 1
(loss / draw / win from the mover's perspective), which plain accuracy-style
objectives do not accept. Judge models on log-loss and calibration, not
accuracy: dice randomness keeps the accuracy ceiling low by construction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import ValueMLP


def train_value_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 256,
) -> ValueMLP:
    """Train a ValueMLP on CPU. Deterministic for a fixed seed."""
    torch.manual_seed(seed)
    model = ValueMLP(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.from_numpy(train_x.astype(np.float32))
    y = torch.from_numpy(train_y.astype(np.float32))
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(x))
        for start in range(0, len(x), batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def evaluate(model: ValueMLP, val_x: np.ndarray, val_y: np.ndarray, bins: int = 10) -> dict:
    """Holdout metrics: log-loss, Brier score, and a calibration table."""
    with torch.no_grad():
        probs = model.predict_proba(torch.from_numpy(val_x.astype(np.float32))).numpy()
    y = val_y.astype(np.float64)
    p = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
    log_loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    calibration = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            calibration.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "count": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "mean_outcome": float(y[mask].mean()),
                }
            )
    return {"log_loss": log_loss, "brier": brier, "calibration": calibration, "n": len(y)}
