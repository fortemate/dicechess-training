import numpy as np
import pytest
import torch

from dicechess_training import train
from dicechess_training.runtime import RunOptions


def test_no_information_baseline_is_ln2_when_balanced():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert train.no_information_log_loss(y) == pytest.approx(np.log(2), abs=1e-9)


def test_no_information_baseline_drops_when_skewed():
    skewed = np.array([1.0] * 9 + [0.0])
    assert train.no_information_log_loss(skewed) < np.log(2)


def test_trained_model_beats_the_baseline_on_learnable_data():
    """A feature that perfectly predicts the label must be learned."""
    rng = np.random.default_rng(0)
    x = rng.random((2_000, 774), dtype=np.float32)
    y = (x[:, 0] > 0.5).astype(np.float32)
    model = train.train_value_model(x, y, epochs=8, seed=0, hidden=32)
    metrics = train.evaluate(model, x, y)
    assert metrics["log_loss"] < train.no_information_log_loss(y)


def test_value_training_uses_adam_without_weight_decay(observed_optimizers):
    x = np.zeros((4, 774), dtype=np.float32)
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    train.train_value_model(x, y, epochs=1, hidden=8, lr=0.002, options=RunOptions(device="cpu"))

    assert len(observed_optimizers) == 1
    optimizer = observed_optimizers[0]
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.defaults["lr"] == 0.002
    assert optimizer.defaults["weight_decay"] == 0
    assert type(optimizer.defaults["weight_decay"]) is int
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0]
