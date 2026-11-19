"""CLI output admission happens before training; trusted run() remains separately tested."""

import sys
from pathlib import Path

import pytest

from dicechess_training import demo


@pytest.fixture
def invocation(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.chdir(root)
    calls = []
    monkeypatch.setattr(demo, "run", lambda **kwargs: calls.append(kwargs))

    def invoke(*args):
        monkeypatch.setattr(sys, "argv", ["demo", *args])
        return demo.main()

    return root, calls, invoke


@pytest.mark.parametrize("choice", ["default", "nested", "current", "absolute", "inside-symlink"])
def test_cli_accepts_output_within_its_working_directory(invocation, choice):
    root, calls, invoke = invocation
    if choice == "default":
        args, expected = [], root / "out"
    elif choice == "nested":
        args, expected = ["--out", "nested/output"], root / "nested/output"
    elif choice == "current":
        args, expected = ["--out", "."], root
    elif choice == "absolute":
        args, expected = ["--out", str(root / "absolute")], root / "absolute"
    else:
        target = root / "target"
        target.mkdir()
        (root / "inside").symlink_to(target, target_is_directory=True)
        args, expected = ["--out", "inside"], target
    assert invoke(*args) is None
    assert len(calls) == 1
    assert Path(calls[0]["out_dir"]).resolve() == expected.resolve()
    assert calls[0]["n_games"] == 400
    assert calls[0]["epochs"] == 2
    assert calls[0]["seed"] == 0
    assert calls[0]["synthetic"] is False
    assert calls[0]["sample_dir"] is None
    assert calls[0]["hidden"] == demo.DEMO_HIDDEN


def test_cli_preserves_other_training_arguments(invocation):
    _, calls, invoke = invocation
    invoke(
        "--games",
        "30",
        "--epochs",
        "1",
        "--seed",
        "8",
        "--synthetic",
        "--sample-dir",
        "fixtures",
        "--hidden",
        "16",
    )
    assert len(calls) == 1
    assert {key: value for key, value in calls[0].items() if key != "out_dir"} == {
        "n_games": 30,
        "epochs": 1,
        "seed": 8,
        "synthetic": True,
        "sample_dir": "fixtures",
        "hidden": 16,
    }
