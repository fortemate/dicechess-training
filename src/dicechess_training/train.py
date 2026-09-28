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
from .runtime import RunOptions, Runtime, identity


def train_value_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 256,
    *,
    options: RunOptions | None = None,
) -> ValueMLP:
    """Train on CPU/CUDA with optional torchrun DDP and epoch-boundary recovery."""
    if min(epochs, batch_size, hidden) < 1 or len(train_x) == 0 or len(train_x) != len(train_y):
        raise ValueError("training needs matching nonempty arrays and positive sizes")
    with Runtime(options or RunOptions()) as runtime:
        runtime.seed(seed)
        model = ValueMLP(hidden=hidden)
        wrapped = runtime.wrap(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        x = torch.from_numpy(train_x.astype(np.float32))
        y = torch.from_numpy(train_y.astype(np.float32))
        signature = identity(
            {
                "trainer": "value-v1",
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "hidden": hidden,
            },
            train_x,
            train_y,
        )
        restored = runtime.restore(model, optimizer, signature)
        start_epoch = restored["epoch"] if restored is not None else 0
        for epoch in range(start_epoch + 1, epochs + 1):
            wrapped.train()
            permutation = torch.randperm(len(x))
            for start in range(0, len(x), batch_size):
                batch, weight = runtime.partition(permutation[start : start + batch_size])
                optimizer.zero_grad()
                loss = loss_fn(wrapped(x[batch].to(runtime.device)), y[batch].to(runtime.device))
                (loss * weight).backward()
                optimizer.step()
            complete = epoch == epochs
            runtime.checkpoint(model, optimizer, signature, {"epoch": epoch, "complete": complete})
            runtime.stop(epoch - start_epoch, complete)
        model.cpu().eval()
        return model


def no_information_log_loss(y: np.ndarray) -> float:
    """Log-loss of always predicting the holdout's own base rate.

    The bar every model must clear: a model scoring worse than this knows less
    than "the average game outcome". For a balanced set it is ln 2 = 0.6931.
    """
    base_rate = float(np.clip(y.astype(np.float64).mean(), 1e-7, 1 - 1e-7))
    labels = y.astype(np.float64)
    return float(-(labels * np.log(base_rate) + (1 - labels) * np.log(1 - base_rate)).mean())


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
