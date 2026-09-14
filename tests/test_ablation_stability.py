"""Numerical training regressions using synthetic, confidently wrong logits."""

import numpy as np
import pytest
import torch

from dicechess_training.ablation import runner


@pytest.mark.parametrize("logit,target", [(100.0, 0.0), (-100.0, 1.0)])
def test_logits_loss_retains_corrective_gradient(logit, target):
    model = runner.ValueMLP(1, [])
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].bias.fill_(logit)
    x = torch.zeros((1, 1))
    y = torch.tensor([[target]])
    torch.nn.BCELoss()(model(x), y).backward()
    assert model.net[0].bias.grad.item() == 0.0
    model.zero_grad()
    loss = torch.nn.BCEWithLogitsLoss()(model.forward_logits(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.net[0].bias.grad.item() == (1.0 if target == 0 else -1.0)


@pytest.mark.parametrize(
    "loss_name,learns", [(None, False), ("bce", False), ("bce-with-logits", True)]
)
def test_training_corrects_saturation_without_changing_legacy_behavior(
    monkeypatch, loss_name, learns
):
    model = runner.ValueMLP(1, [])
    with torch.no_grad():
        model.net[0].weight.fill_(100.0)
        model.net[0].bias.zero_()
    monkeypatch.setattr(runner, "ValueMLP", lambda *_: model)
    cfg = {"hidden_dims": [], "learning_rate": 1.0, "batch_size": 4, "epochs": 2}
    if loss_name is not None:
        cfg["loss"] = loss_name
    trained = runner.train_model(np.ones((4, 1), np.float32), np.zeros(4, np.float32), 1, cfg, 7)
    weight = trained.net[0].weight.item()
    assert weight < 99.0 if learns else weight == 100.0
    assert not trained.training


def test_probability_inference_and_checkpoint_layout_are_preserved():
    model = runner.ValueMLP(3, [4])
    restored = runner.ValueMLP(3, [4])
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.tensor([[1.0, -2.0, 3.0]])
    assert torch.equal(model(x), torch.sigmoid(model.forward_logits(x)))
    assert torch.equal(model(x), restored(x))
    assert set(model.state_dict()) == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}


@pytest.mark.parametrize("loss_name", ["typo", None, 123])
def test_unknown_loss_fails_closed(loss_name):
    with pytest.raises(ValueError, match="unsupported training loss"):
        runner.train_model(np.zeros((1, 1)), np.zeros(1), 1, {"loss": loss_name}, 7)
