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


@pytest.mark.parametrize(
    "choice", ["parent", "absolute", "prefix-sibling", "directory-symlink", "directory-loop"]
)
def test_cli_refuses_escape_before_creating_output_or_training(invocation, capsys, choice):
    root, calls, invoke = invocation
    outside = root.parent / "outside"
    if choice == "parent":
        output = "../outside/new"
    elif choice == "absolute":
        output = str(outside / "new")
    elif choice == "prefix-sibling":
        output = str(root.with_name(root.name + "-other") / "new")
    elif choice == "directory-symlink":
        outside.mkdir()
        (root / "link").symlink_to(outside, target_is_directory=True)
        output = "link/new"
    else:
        (root / "loop").symlink_to("loop", target_is_directory=True)
        output = "loop/new"
    with pytest.raises(SystemExit) as caught:
        invoke("--out", output, "--synthetic")
    assert caught.value.code == 2
    assert calls == []
    assert not (outside / "new").exists()
    assert not root.with_name(root.name + "-other").exists()
    assert "output" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("source", ["synthetic", "sample"])
@pytest.mark.parametrize("extension", ["onnx", "onnx.data"])
def test_cli_refuses_existing_model_symlink_outside_workspace(
    invocation, capsys, source, extension
):
    root, calls, invoke = invocation
    outside = root.parent / "existing.onnx"
    outside.write_bytes(b"preserve existing model")
    output = root / "out"
    output.mkdir()
    (output / f"value-{source}.{extension}").symlink_to(outside)
    args = ["--out", "out"]
    if source == "synthetic":
        args.append("--synthetic")
    with pytest.raises(SystemExit) as caught:
        invoke(*args)
    assert caught.value.code == 2
    assert calls == []
    assert outside.read_bytes() == b"preserve existing model"
    assert "output" in capsys.readouterr().err.lower()


def test_cli_refuses_unresolvable_output_before_training(invocation, monkeypatch, capsys):
    _, calls, invoke = invocation
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path.name == "broken-output":
            raise OSError("authored resolution failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(SystemExit) as caught:
        invoke("--out", "broken-output")
    assert caught.value.code == 2
    assert calls == []
    assert "output" in capsys.readouterr().err.lower()
