"""Recovery preserves training state; distributed uneven batches preserve the objective."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dicechess_training.runtime import (
    RunOptions,
    Runtime,
    TrainingPaused,
    TrainingRuntimeError,
    atomic_save,
)
from dicechess_training.train import train_value_model

ROOT = Path(__file__).resolve().parents[1]


def _values():
    x = np.random.default_rng(3).normal(size=(9, 774)).astype(np.float32)
    return x, np.linspace(0, 1, len(x), dtype=np.float32)


def test_resume_refuses_changed_inputs_configuration_or_world_size(tmp_path):
    x, y = _values()
    checkpoint = tmp_path / "checkpoint.pt"
    with pytest.raises(TrainingPaused):
        train_value_model(
            x,
            y,
            epochs=3,
            hidden=8,
            options=RunOptions(device="cpu", checkpoint=checkpoint, stop_after_epochs=1),
        )
    options = RunOptions(device="cpu", resume=checkpoint)
    changed = x.copy()
    changed[0, 0] += 1
    for data, hidden in ((changed, 8), (x, 9)):
        with pytest.raises(TrainingRuntimeError, match="do not match"):
            train_value_model(data, y, epochs=3, hidden=hidden, options=options)
    payload = torch.load(checkpoint, weights_only=True)
    payload["environment"]["world_size"] = 2
    atomic_save(payload, checkpoint)
    with pytest.raises(TrainingRuntimeError, match="do not match"):
        train_value_model(x, y, epochs=3, hidden=8, options=options)


def test_failed_atomic_save_preserves_last_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    atomic_save({"epoch": 1}, checkpoint)

    def fail(*_):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_save({"epoch": 2}, checkpoint)
    assert torch.load(checkpoint, weights_only=True) == {"epoch": 1}
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_bounded_run_requires_a_destination():
    with pytest.raises(TrainingRuntimeError, match="checkpoint"):
        RunOptions(stop_after_epochs=1)


@pytest.mark.parametrize("processes", [1, 2])
def test_actual_trainers_resume_exactly_in_new_processes(tmp_path, processes):
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
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}
    for mode in ("full", "segment", "resume"):
        result = subprocess.run(
            [
                *launcher,
                str(ROOT / "scripts/check_training_runtime.py"),
                str(tmp_path),
                mode,
                "--device",
                "cpu",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    if processes == 2:
        serial_dir = tmp_path / "serial"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_training_runtime.py"),
                str(serial_dir),
                "full",
                "--device",
                "cpu",
            ],
            env=environment,
            check=True,
            capture_output=True,
            timeout=60,
        )
        for trainer in ("value", "prerank"):
            first = torch.load(tmp_path / trainer / "full/result-rank-0.pt", weights_only=True)
            second = torch.load(tmp_path / trainer / "full/result-rank-1.pt", weights_only=True)
            serial = torch.load(serial_dir / trainer / "full/result-rank-0.pt", weights_only=True)
            # Same global batches/objective, allowing floating-point reduction-order differences.
            for key in first["model"]:
                assert torch.equal(first["model"][key], second["model"][key])
                torch.testing.assert_close(
                    first["model"][key], serial["model"][key], rtol=2e-5, atol=2e-6
                )


def test_completed_early_stop_resumes_without_another_update(tmp_path, monkeypatch):
    from dicechess_training.prerank import train as trainer
    from test_prerank_training import _corpus

    corpus = _corpus([5] * 6, ["train"] * 4 + ["validation"] * 2)
    monkeypatch.setattr(trainer, "ranking_metrics", lambda *args: {"recall_at_k": 0.5})
    hyper = trainer.Hyperparameters(max_epochs=8, patience=1, k=2)
    checkpoint = tmp_path / "checkpoint.pt"
    report, model, scores = trainer.train(
        corpus, hyper, options=RunOptions(device="cpu", checkpoint=checkpoint)
    )
    assert len(report["history"]) == 2
    resumed, restored, actual = trainer.train(
        corpus, hyper, options=RunOptions(device="cpu", resume=checkpoint)
    )
    assert report == resumed
    assert np.array_equal(scores, actual)
    assert all(
        torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()
    )


@pytest.mark.parametrize("failure", [OSError, RuntimeError])
def test_checkpoint_writer_errors_become_coordinated_failures(tmp_path, monkeypatch, failure):
    def fail(*_):
        raise failure("writer failed")

    monkeypatch.setattr(torch, "save", fail)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    with Runtime(RunOptions(device="cpu", checkpoint=tmp_path / "checkpoint.pt")) as runtime:
        with pytest.raises(TrainingRuntimeError, match="checkpoint save failed"):
            runtime.checkpoint(model, optimizer, "synthetic", {"epoch": 1})
