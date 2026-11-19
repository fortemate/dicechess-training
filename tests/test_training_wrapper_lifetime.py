"""Training wrappers finish before their execution context releases native resources."""

import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dicechess_training.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkers():
    result = {}
    for name in ("training", "experimental"):
        path = ROOT / "scripts" / f"check_{name}_runtime.py"
        spec = importlib.util.spec_from_file_location(f"lifetime_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[name] = module
    return result


class Lease(nn.Module):
    """A separate resource handle around the real model, preserving its computations."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@pytest.mark.parametrize("trainer", ["value", "prerank", "A", "B", "C"])
@pytest.mark.parametrize("mode", ["full", "segment"])
def test_wrapper_is_released_before_context_closes(checkers, monkeypatch, tmp_path, trainer, mode):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    previous_threads = torch.get_num_threads()
    checker = checkers["training" if trainer in ("value", "prerank") else "experimental"]
    selection = {"trainer": trainer} if trainer in ("value", "prerank") else {"arm": trainer}

    def run(folder):
        checker.run(
            SimpleNamespace(
                destination=folder, mode=mode, device="cpu", interrupt=False, **selection
            )
        )

    references = []
    live_at_close = []
    original_wrap = Runtime.wrap
    original_exit = Runtime.__exit__

    def wrap(context, model):
        leased = Lease(original_wrap(context, model))
        references.append(weakref.ref(leased))
        return leased

    def close(context, *error):
        live_at_close.append(any(reference() is not None for reference in references))
        return original_exit(context, *error)

    baseline_root = tmp_path / "baseline"
    observed_root = tmp_path / "observed"
    try:
        run(baseline_root)
        monkeypatch.setattr(Runtime, "wrap", wrap)
        monkeypatch.setattr(Runtime, "__exit__", close)
        run(observed_root)
        baseline = sorted(path.relative_to(baseline_root) for path in baseline_root.rglob("*.pt"))
        observed = sorted(path.relative_to(observed_root) for path in observed_root.rglob("*.pt"))
        assert baseline == observed
        assert len(baseline) == (2 if mode == "full" else 1)
        for name in baseline:
            expected = torch.load(baseline_root / name, weights_only=True, map_location="cpu")
            actual = torch.load(observed_root / name, weights_only=True, map_location="cpu")
            assert checkers["training"].equal(expected, actual)
        assert len(references) == 1
        assert live_at_close == [False]
    finally:
        torch.set_num_threads(previous_threads)
