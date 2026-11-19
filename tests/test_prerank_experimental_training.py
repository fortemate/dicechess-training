"""Exact rule bypass, early-stop isolation and actual fresh-process recovery."""

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dicechess_training.prerank import experimental_train as trainer
from dicechess_training.prerank.experimental import digest, make_model
from dicechess_training.prerank.model import listwise_loss
from dicechess_training.runtime import RunOptions, TrainingPaused, TrainingRuntimeError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_experimental_runtime.py"
spec = importlib.util.spec_from_file_location("experimental_oracle", SCRIPT)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_terminal_branch_matches_explicit_full_probability_cross_entropy():
    corpus = oracle.fixture()
    prepared = trainer.prepare(corpus)
    group = 0
    lo, hi = corpus.offsets[group : group + 2]

    class Scores(nn.Module):
        def forward(self, features):
            assert len(features) == 2
            return features[:, :1]

    loss = trainer.batch_loss(
        Scores(), corpus, prepared, np.array([group]), "A", torch.device("cpu")
    )
    gains = trainer.target_gains(corpus.targets[lo:hi])
    mass = gains[1:].sum()
    probabilities = np.concatenate(
        (gains[:1], mass * torch.softmax(torch.from_numpy(corpus.a[lo + 1 : hi, 0]), 0).numpy())
    )
    expected = -np.sum(gains * np.log(probabilities))
    assert float(loss) == pytest.approx(expected, rel=1e-6)


def test_all_terminal_group_has_constant_loss_and_zero_model_gradient():
    corpus = oracle.fixture()
    prepared = trainer.prepare(corpus)
    for arm in ("A", "B", "C"):
        model = make_model(corpus, arm)
        loss = trainer.batch_loss(model, corpus, prepared, np.array([8]), arm, torch.device("cpu"))
        assert float(loss.detach()) == pytest.approx(np.log(2))
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())
        assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_screen_never_changes_fitting_and_each_arm_sees_the_same_group_order():
    corpus = oracle.fixture()
    changed = oracle.fixture()
    lo = corpus.offsets[-2]
    changed.a[lo:] += 100
    changed.b[lo:, :9] += 100
    changed.targets[lo:] += 100
    hyper = trainer.Hyperparameters(max_epochs=2, patience=2, batch_groups=4)
    first, model = trainer.train(corpus, "A", hyper, options=RunOptions(device="cpu"))
    second, again = trainer.train(changed, "A", hyper, options=RunOptions(device="cpu"))
    assert first["history"] == second["history"]
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in model.state_dict().items())
    orders = [x["order_sha256"] for x in first["history"]]
    for arm in ("B", "C"):
        report, _ = trainer.train(corpus, arm, hyper, options=RunOptions(device="cpu"))
        assert [x["order_sha256"] for x in report["history"]] == orders
        assert [x["updates"] for x in report["history"]] == [3, 3]


def test_completed_early_stop_checkpoint_does_not_update_on_resume(tmp_path, monkeypatch):
    corpus = oracle.fixture()
    checkpoint = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(trainer, "validation_loss", lambda *args: 1.0)
    hyper = trainer.Hyperparameters(max_epochs=8, patience=1, batch_groups=4)
    report, model = trainer.train(
        corpus, "C", hyper, options=RunOptions(device="cpu", checkpoint=checkpoint)
    )
    assert len(report["history"]) == 2
    again, restored = trainer.train(
        corpus, "C", hyper, options=RunOptions(device="cpu", resume=checkpoint)
    )
    assert again == report
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_resume_refuses_other_arm_changed_inputs_and_plan(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    corpus = oracle.fixture()
    hyper = trainer.Hyperparameters(max_epochs=3, patience=3)
    segment_options = RunOptions(device="cpu", checkpoint=checkpoint, stop_after_epochs=1)
    resume_options = RunOptions(device="cpu", resume=checkpoint)
    with pytest.raises(TrainingPaused):
        trainer.train(
            corpus,
            "A",
            hyper,
            options=segment_options,
            input_identity="first",
        )
    changed = oracle.fixture()
    changed.tie_order[:3] = changed.tie_order[:3][::-1]
    for data, arm, plan in (
        (corpus, "B", "first"),
        (changed, "A", "first"),
        (corpus, "A", "other"),
    ):
        with pytest.raises(TrainingRuntimeError, match="do not match"):
            trainer.train(
                data,
                arm,
                hyper,
                options=resume_options,
                input_identity=plan,
            )


@pytest.mark.parametrize("processes", [1, 2])
def test_actual_all_arm_recovery_in_fresh_processes(tmp_path, processes):
    launcher = [sys.executable]
    if processes == 2:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        launcher += [
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
        ]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}
    for mode in ("full", "segment", "resume"):
        result = subprocess.run(
            [*launcher, str(SCRIPT), str(tmp_path), mode],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    if processes == 2:
        serial = tmp_path / "serial"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(serial), "full"],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for arm in ("A", "B", "C"):
            first = torch.load(tmp_path / arm / "full/result-rank-0.pt", weights_only=True)
            other = torch.load(tmp_path / arm / "full/result-rank-1.pt", weights_only=True)
            reference = torch.load(serial / arm / "full/result-rank-0.pt", weights_only=True)
            for key, value in first["model"].items():
                assert torch.equal(value, other["model"][key])
                if not key.endswith("bias"):
                    torch.testing.assert_close(value, reference["model"][key], rtol=3e-5, atol=3e-6)
            # Cross-topology FP32 training need not be bitwise identical. Compare the learned
            # conditional probabilities and exact order; same-topology recovery above remains
            # exact for EVERY parameter.
            data = oracle.fixture()
            model, serial_model = make_model(data, arm), make_model(data, arm)
            model.load_state_dict(first["model"])
            serial_model.load_state_dict(reference["model"])
            features = {"A": data.a, "B": data.b, "C": data.c}[arm]
            with torch.no_grad():
                scores = model(torch.from_numpy(features)).flatten()
                expected = serial_model(torch.from_numpy(features)).flatten()
            for lo, hi in zip(data.offsets[:-1], data.offsets[1:], strict=True):
                rows = np.arange(lo, hi)[~data.terminal[lo:hi]]
                if len(rows):
                    _assert_ranking_close(scores[rows], expected[rows])


def _assert_ranking_close(scores, reference):
    assert torch.isfinite(scores).all()
    assert torch.isfinite(reference).all()
    # Compare the quantity used by the listwise objective. Only this normalization is FP64;
    # the original FP32 forward, checkpoint and parameter comparisons remain unchanged.
    # Softmax's Jacobian row absolute sum is 2*p*(1-p) <= 1/2, so half the former
    # 6e-6 centered-logit absolute floor gives a 3e-6 probability budget, with no rtol.
    probabilities = scores.to(torch.float64).softmax(dim=0)
    reference_probabilities = reference.to(torch.float64).softmax(dim=0)
    torch.testing.assert_close(probabilities, reference_probabilities, rtol=0, atol=3e-6)
    assert torch.equal(scores.argsort(stable=True), reference.argsort(stable=True))


@pytest.mark.parametrize(
    ("scores", "reference"),
    [
        ([0.0, 0.25, 0.5], [0.0, 0.25, 0.5]),
        ([8.0, 8.25, 8.5], [0.0, 0.25, 0.5]),
        ([0.0, 0.0, 0.25], [1.0, 1.0, 1.25]),
        ([100.0], [-100.0]),
    ],
)
def test_ranking_oracle_accepts_identical_order_and_group_shifts(scores, reference):
    _assert_ranking_close(torch.tensor(scores), torch.tensor(reference))


@pytest.mark.parametrize(
    ("scores", "reference"),
    [
        ([0.0, 0.25, 0.6], [0.0, 0.25, 0.5]),
        ([0.0, 0.5, 1.0], [0.0, 0.25, 0.5]),
        ([1e-8, 0.0], [0.0, 1e-8]),
        ([float("nan"), 0.0], [0.0, 0.0]),
        ([float("inf"), 0.0], [0.0, 0.0]),
        ([float("-inf"), 0.0], [0.0, 0.0]),
        ([float("-inf"), 0.0], [-20.0, 0.0]),
        ([-20.0, 0.0], [float("-inf"), 0.0]),
    ],
)
def test_ranking_oracle_rejects_distortion_order_changes_and_nonfinite(scores, reference):
    actual = torch.tensor(scores)
    expected = torch.tensor(reference)
    with pytest.raises(AssertionError):
        _assert_ranking_close(actual, expected)


@pytest.mark.parametrize(
    ("amplitude", "candidates", "accepted"),
    [(5e-6, 2, True), (7e-6, 2, False), (7e-6, 3, True)],
)
def test_listwise_parity_uses_probability_error_with_exact_order(amplitude, candidates, accepted):
    # For two candidates, probabilities differ from 1/2 by tanh(amplitude)/2.
    # With three near-uniform candidates the maximum change is about amplitude/3.
    # These are authored logits, independent of any trained checkpoint.
    actual = torch.linspace(-amplitude, amplitude, candidates, dtype=torch.float64)
    expected = torch.zeros(candidates, dtype=torch.float64)
    if accepted:
        _assert_ranking_close(actual, expected)
    else:
        with pytest.raises(AssertionError):
            _assert_ranking_close(actual, expected)


@pytest.mark.parametrize("shifts", [[0.0, 0.0], [8.0, -16.0]])
def test_listwise_objective_has_analytic_probabilities_and_independent_group_shifts(shifts):
    # Conditional probabilities are [1/4, 3/4] and [4/5, 1/5]. Each target
    # assigns all mass to one candidate; the two groups have equal weight.
    scores = torch.tensor([0.0, np.log(3.0), np.log(4.0), 0.0], dtype=torch.float64)
    group_index = torch.tensor([0, 0, 1, 1])
    scores = (scores + torch.tensor(shifts, dtype=torch.float64)[group_index]).requires_grad_()
    gains = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    loss = listwise_loss(scores, gains, group_index, 2)
    assert float(loss.detach()) == pytest.approx((np.log(4.0) + np.log(5.0)) / 2)
    loss.backward()
    torch.testing.assert_close(
        scores.grad,
        torch.tensor([-3 / 8, 3 / 8, 2 / 5, -2 / 5], dtype=torch.float64),
    )


def test_real_process_termination_after_checkpoint_recovers(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}

    def run(mode, interrupt=False):
        args = [sys.executable, str(SCRIPT), str(tmp_path), mode, "--arm", "C"]
        return subprocess.run(
            args + (["--interrupt"] if interrupt else []),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert run("full").returncode == 0
    assert run("segment", True).returncode == 42
    result = run("resume")
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_requires_pinned_plan_and_safe_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input"
    source.mkdir()
    data = oracle.fixture()
    from dicechess_training.prerank.experimental import ARRAY_KEYS, FEATURE_SCHEMA, SCHEMA

    np.savez_compressed(source / "arrays.npz", **{k: getattr(data, k) for k in ARRAY_KEYS})
    manifest = {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "engine_version": "0.14.0",
        "arrays_sha256": digest(source / "arrays.npz"),
        "groups": data.groups,
        "candidates": len(data.targets),
        "split_groups": {s: len(data.of(s)) for s in ("train", "early_stop", "screen")},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    mh = digest(source / "manifest.json")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": trainer.PROTOCOL,
                "input_manifest_sha256": mh,
                "arm": "A",
                "hyperparameters": {"max_epochs": 3, "patience": 3, "batch_groups": 4},
            }
        )
    )
    output = tmp_path / "run"
    args = [
        str(source),
        str(plan),
        str(output),
        "--manifest-sha256",
        mh,
        "--plan-sha256",
        digest(plan),
        "--device",
        "cpu",
    ]
    assert trainer.main(args + ["--stop-after-epochs", "1"]) == 75
    assert not (output / trainer.RESULT_FILE).exists()
    assert trainer.main(args + ["--resume"]) == 0
    assert (output / trainer.RESULT_FILE).exists()
    assert trainer.main(args) == 1
    assert trainer.main(args[:-3] + ["0" * 64, "--device", "cpu", "--resume"]) == 1


@pytest.mark.parametrize("groups", [[0, 1, 8], [0], [8]])
def test_uneven_rank_partitions_preserve_full_batch_loss_and_gradients(groups):
    data = oracle.fixture()
    prepared = trainer.prepare(data)
    groups = np.asarray(groups)
    for arm in ("A", "B", "C"):
        model = make_model(data, arm)
        loss = trainer.batch_loss(model, data, prepared, groups, arm, torch.device("cpu"))
        loss.backward()
        expected_gradients = [p.grad.clone() for p in model.parameters()]
        gradients = [torch.zeros_like(g) for g in expected_gradients]
        total = 0.0
        for rank in range(2):
            local = groups[rank::2]
            weight = len(local) * 2 / len(groups)
            local = local if len(local) else groups[:1]
            worker = copy.deepcopy(model)
            worker.zero_grad()
            local_loss = (
                trainer.batch_loss(worker, data, prepared, local, arm, torch.device("cpu")) * weight
            )
            local_loss.backward()
            total += float(local_loss.detach()) / 2
            for accumulated, parameter in zip(gradients, worker.parameters(), strict=True):
                accumulated += parameter.grad / 2
        assert total == pytest.approx(float(loss.detach()), rel=1e-6)
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-7)
