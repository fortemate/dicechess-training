"""Graph-embedded probability calibration.

The contract these tests defend is that the calibration is part of the artifact rather than a
setting a consumer applies: it must survive the ONNX export, it must leave an uncalibrated model
byte-identical to what the packager produced before calibration existed, and it must be selected
from rows the model was not fitted on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dicechess_training.ablation.export import export_candidate
from dicechess_training.ablation.runner import (
    TEMPERATURE_GRID_MAX,
    TEMPERATURE_GRID_MIN,
    ValueMLP,
    fit_logit_temperature,
)
from dicechess_training.ablation.runner import predict as torch_predict
from dicechess_training.benchmark import core
from dicechess_training.candidate import CandidateConfig, build_candidate
from dicechess_training.candidate.build import MODEL_FILE, CandidateError
from dicechess_training.contracts import kcp13

FIXTURE = core.ROOT / "tests/fixtures/benchmark"


def _model(temperature=None):
    torch.manual_seed(7)
    return ValueMLP(kcp13.FEATURE_COUNT, [8, 8], logit_temperature=temperature)


def _rows(n=64, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, kcp13.FEATURE_COUNT)).astype(np.float32)


def test_an_uncalibrated_model_is_unchanged():
    """Protocols that never asked for calibration must keep producing the same numbers."""
    model = _model()
    assert not model.calibrated
    assert "logit_temperature" not in dict(model.named_buffers())
    x = torch.tensor(_rows())
    assert torch.equal(model(x), model.net(model._standardise(x)))


def test_calibration_divides_the_logit_before_the_sigmoid():
    x = torch.tensor(_rows())
    model = _model()
    logits = model.forward_logits(x)
    model.set_logit_temperature(1.25)
    assert torch.allclose(model(x), torch.sigmoid(logits / 1.25), atol=1e-6)
    # Training scores stay uncalibrated: the constant is fitted after training, against a holdout.
    assert torch.equal(model.forward_logits(x), logits)


def test_temperature_is_a_buffer_so_it_travels_with_the_checkpoint():
    model = _model(temperature=1.5)
    assert "logit_temperature" in dict(model.named_buffers())
    assert float(model.state_dict()["logit_temperature"]) == pytest.approx(1.5)


def test_resetting_the_temperature_does_not_register_a_second_buffer():
    model = _model(temperature=1.5)
    model.set_logit_temperature(1.1)
    assert float(model.logit_temperature) == pytest.approx(1.1)
    assert [name for name, _ in model.named_buffers()].count("logit_temperature") == 1


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_temperature_is_refused(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        _model().set_logit_temperature(bad)


def test_calibration_survives_the_onnx_export(tmp_path):
    """The whole point: the served graph carries the calibration, not the caller."""
    model = _model(temperature=1.37)
    path = export_candidate(model, kcp13.SCHEMA_ID, tmp_path / MODEL_FILE)
    x = _rows(n=48, seed=9)
    torch_p = torch_predict(model, x)
    onnx_p = kcp13.predict(path, x)
    assert np.max(np.abs(torch_p - onnx_p)) <= 1e-6
    # And it is genuinely calibrated rather than an exported identity.
    uncalibrated = kcp13.predict(
        export_candidate(_model(), kcp13.SCHEMA_ID, tmp_path / "raw.onnx"), x
    )
    assert np.max(np.abs(onnx_p - uncalibrated)) > 1e-4


def test_fitting_needs_a_nonempty_holdout():
    with pytest.raises(ValueError, match="non-empty"):
        fit_logit_temperature(
            _model(), np.empty((0, kcp13.FEATURE_COUNT), np.float32), np.array([])
        )


def test_an_unsupported_rule_is_refused():
    with pytest.raises(ValueError, match="unsupported calibration rule"):
        fit_logit_temperature(_model(), _rows(), np.zeros(64), rule="accuracy")


def test_fitting_stays_inside_the_declared_grid_and_reports_both_ends():
    model = _model()
    x = _rows(n=400, seed=5)
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=len(x)).astype(float)
    result = fit_logit_temperature(model, x, y, rule="brier")
    assert TEMPERATURE_GRID_MIN <= result["temperature"] <= TEMPERATURE_GRID_MAX
    assert result["rule"] == "brier"
    assert result["holdout_rows"] == len(x)
    # Both ends are reported so a run can be judged on what the calibration bought.
    assert set(result["selected"]) == {"log_loss", "brier", "ece"}
    assert result["selected"]["brier"] <= result["uncalibrated"]["brier"]


def test_an_already_calibrated_model_keeps_its_temperature_near_one():
    """Ties and flat optima break towards the identity, so an uninformative holdout buys nothing."""
    model = _model()
    x = _rows(n=2000, seed=21)
    with torch.no_grad():
        p = torch.sigmoid(model.forward_logits(torch.tensor(x))).squeeze(-1).numpy()
    y = np.random.default_rng(4).binomial(1, p).astype(float)
    assert fit_logit_temperature(model, x, y, rule="log_loss")["temperature"] == pytest.approx(
        1.0, abs=0.25
    )


def test_the_rule_is_digested_into_the_config_so_it_cannot_be_chosen_afterwards():
    base = CandidateConfig(seed=1, model_id="m")
    other = CandidateConfig(seed=1, model_id="m", probability_calibration="log_loss")
    assert core.digest(base.as_record()) != core.digest(other.as_record())
    assert base.as_record()["probability_calibration"] == "brier"


def test_an_unknown_rule_never_produces_a_package(tmp_path):
    with pytest.raises(CandidateError, match="unsupported probability calibration"):
        build_candidate(
            FIXTURE,
            tmp_path / "package",
            CandidateConfig(seed=1, model_id="m", probability_calibration="auc"),
        )
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize("rule", ["none", "brier"])
def test_a_built_package_records_its_calibration_and_never_declares_it(tmp_path, rule):
    directory = tmp_path / rule
    summary = build_candidate(
        FIXTURE,
        directory,
        CandidateConfig(seed=11, model_id=f"cal-{rule}", probability_calibration=rule),
    )
    provenance = summary["manifest"]["provenance"]
    assert provenance["probability_calibration"] == rule
    # ADR 0001: the manifest may never carry a temperature the evaluator would have to apply.
    assert summary["manifest"].get("calibration", {}) in ({}, {"temperature": 1.0})
    if rule == "none":
        assert provenance["logit_temperature"] == "1.0"
        assert summary["calibration"] is None
    else:
        assert float(provenance["logit_temperature"]) > 0.0
        assert summary["calibration"]["rule"] == "brier"
        assert summary["calibration"]["holdout_rows"] == summary["inner_tuning_rows"]
    # Whatever the rule, the package is still the artifact the benchmark agrees to read.
    data_manifest, rows = core.load_dataset(FIXTURE)
    core.load_candidate(directory, data_manifest, core.load_protocol())
    assert (
        len(
            kcp13.predict(
                directory / MODEL_FILE,
                np.asarray([r["features"] for r in rows[:16]], dtype=np.float32),
            )
        )
        == 16
    )
