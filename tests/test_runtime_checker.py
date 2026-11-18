"""The recovery checker's orchestration contract, independently of trainer internals."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_training_runtime.py"


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("runtime_checker_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(destination, mode="full", trainer=None):
    return SimpleNamespace(
        destination=destination, mode=mode, trainer=trainer, device="cpu", interrupt=False
    )


@pytest.mark.parametrize("selection", [None, "value", "prerank"])
@pytest.mark.parametrize("mode", ["full", "segment", "resume"])
@pytest.mark.parametrize("rank", [0, 1])
def test_trainer_selection_options_artifacts_and_primary_output(
    checker, tmp_path, monkeypatch, capsys, selection, mode, rank
):
    monkeypatch.setenv("RANK", str(rank))
    events = []
    states = {
        name: {"weight": torch.tensor([index])} for index, name in enumerate(("value", "prerank"))
    }
    report = {"synthetic": "report"}
    scores = np.array([0.25, 0.75], dtype=np.float32)
    calls = []

    def trained(name, options):
        events.append(("train", name))
        calls.append((name, options))
        if mode == "segment":
            raise checker.TrainingPaused()
        return SimpleNamespace(state_dict=lambda: states[name])

    def value(x, y, *, epochs, batch_size, hidden, options):
        assert x.shape == (9, 774)
        assert x.dtype == np.float32
        assert y.shape == (9,)
        assert y.dtype == np.float32
        np.testing.assert_array_equal(y, (x[:, 0] + 1) / 2)
        assert (epochs, batch_size, hidden) == (3, 4, 8)
        return trained("value", options)

    def prerank(corpus, hyper, *, options):
        assert corpus.features.shape == (58, 9)
        assert corpus.offsets.tolist() == [0, 3, 10, 15, 19, 25, 28, 36, 40, 45, 51, 58]
        assert (
            hyper.hidden_dims,
            hyper.max_epochs,
            hyper.patience,
            hyper.batch_groups,
            hyper.k,
        ) == ((8,), 3, 3, 4, 2)
        return report, trained("prerank", options), scores

    saved = []

    def save(payload, path):
        events.append(("save", path))
        saved.append((payload, path))

    def load(path, *, weights_only, map_location):
        assert weights_only is True
        assert map_location == "cpu"
        events.append(("load", path))
        return {"nested": [torch.tensor([1, 2]), {"complete": True}]}

    monkeypatch.setattr(checker, "train_value_model", value)
    monkeypatch.setattr(checker, "train", prerank)
    monkeypatch.setattr(checker, "atomic_save", save)
    monkeypatch.setattr(checker.torch, "load", load)
    monkeypatch.setattr(checker.torch, "set_num_threads", lambda n: events.append(("threads", n)))
    checker.run(_args(tmp_path, mode, selection))

    names = [selection] if selection else ["value", "prerank"]
    expected_events = [("threads", 1)]
    for name, options in calls:
        path = tmp_path / name
        folder = path / ("full" if mode == "full" else "resumed")
        assert options.device == "cpu"
        assert options.checkpoint == folder / "checkpoint.pt"
        assert options.resume == (folder / "checkpoint.pt" if mode == "resume" else None)
        assert options.stop_after_epochs == (1 if mode == "segment" else None)
        expected_events.append(("train", name))
        if mode != "segment":
            expected_events.append(("save", folder / f"result-rank-{rank}.pt"))
        if mode == "resume":
            for filename in ("checkpoint.pt", f"result-rank-{rank}.pt"):
                expected_events.extend(
                    [("load", path / "full" / filename), ("load", path / "resumed" / filename)]
                )
    assert [name for name, _ in calls] == names
    assert events == expected_events
    assert len(saved) == (0 if mode == "segment" else len(names))
    for (payload, _), name in zip(saved, names, strict=mode != "segment"):
        assert payload["model"] is states[name]
        assert set(payload) == ({"model"} if name == "value" else {"model", "report", "scores"})
        if name == "prerank":
            assert payload["report"] is report
            np.testing.assert_array_equal(payload["scores"].numpy(), scores)
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line) for line in lines] == (
        [
            {
                "trainer": name,
                "mode": mode,
                "complete": mode != "segment",
                "resume_equal": mode == "resume",
            }
            for name in names
        ]
        if rank == 0
        else []
    )


@pytest.mark.parametrize("mode", ["full", "resume"])
def test_unexpected_training_pause_propagates(checker, tmp_path, monkeypatch, capsys, mode):
    failure = checker.TrainingPaused("synthetic pause")

    def pause(*args, **kwargs):
        raise failure

    monkeypatch.setattr(checker, "train_value_model", pause)
    with pytest.raises(checker.TrainingPaused) as caught:
        checker.run(_args(tmp_path, mode, "value"))
    assert caught.value is failure
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("stage", ["train", "save", "load"])
def test_errors_stop_orchestration_without_success_output(
    checker, tmp_path, monkeypatch, capsys, stage
):
    failure = OSError("synthetic failure")

    def fail(*args, **kwargs):
        raise failure

    model = SimpleNamespace(state_dict=lambda: {})
    monkeypatch.setattr(checker, "train_value_model", lambda *a, **k: model)
    monkeypatch.setattr(checker, "atomic_save", lambda *a: None)
    target, attribute = {
        "train": (checker, "train_value_model"),
        "save": (checker, "atomic_save"),
        "load": (checker.torch, "load"),
    }[stage]
    monkeypatch.setattr(target, attribute, fail)
    with pytest.raises(OSError) as caught:
        checker.run(_args(tmp_path, "resume", "value"))
    assert caught.value is failure
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("filename", ["checkpoint.pt", "result-rank-0.pt"])
def test_resume_compares_checkpoint_then_result_and_rejects_mismatch(
    checker, tmp_path, monkeypatch, capsys, filename
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        checker, "train_value_model", lambda *a, **k: SimpleNamespace(state_dict=dict)
    )
    monkeypatch.setattr(checker, "atomic_save", lambda *a: None)
    loaded = []

    def load(path, **kwargs):
        loaded.append(path)
        return torch.tensor([int(path.parent.name == "resumed" and path.name == filename)])

    monkeypatch.setattr(checker.torch, "load", load)
    with pytest.raises(AssertionError, match=f"value: resumed {filename} differs"):
        checker.run(_args(tmp_path, "resume", "value"))
    expected = ["checkpoint.pt"] if filename == "checkpoint.pt" else ["checkpoint.pt", filename]
    assert loaded == [
        tmp_path / "value" / folder / name for name in expected for folder in ("full", "resumed")
    ]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("trainer", ["value", "prerank"])
def test_interrupt_exits_only_after_a_loadable_first_epoch_checkpoint(tmp_path, trainer):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            "full",
            "--device",
            "cpu",
            "--trainer",
            trainer,
            "--interrupt",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "RANK": "0", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 42, completed.stdout + completed.stderr
    checkpoint = torch.load(tmp_path / trainer / "full/checkpoint.pt", weights_only=True)
    assert checkpoint["progress"]["epoch"] == 1
    assert checkpoint["progress"]["complete"] is False
    assert not (tmp_path / trainer / "full/result-rank-0.pt").exists()
    assert completed.stdout == ""
