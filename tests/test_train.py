import numpy as np
import pytest

from dicechess_training import train


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
