"""Synthetic CPU/CUDA/DDP interruption oracle for all three experimental arms."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from dicechess_training.prerank.experimental import ExperimentalCorpus
from dicechess_training.prerank.experimental_train import Hyperparameters, train
from dicechess_training.runtime import RunOptions, Runtime, TrainingPaused, atomic_save


def fixture() -> ExperimentalCorpus:
    sizes = np.array([3, 5, 4, 6, 3, 4, 7, 5, 2, 3, 4, 5])
    offsets = np.concatenate(([0], sizes.cumsum())).astype(np.int64)
    n = int(offsets[-1])
    a = np.sin(np.arange(n * 9).reshape(-1, 9) / 13).astype(np.float32)
    b = np.column_stack([a, np.zeros((n, 4), np.float32)])
    targets = np.rint(a[:, 0] * 100).astype(np.int64)
    terminal = np.zeros(n, np.bool_)
    terminal[offsets[[0, 2, 4, 6, 9]]] = True
    terminal[offsets[8] : offsets[9]] = True
    targets[terminal] = 2**31 - 1
    c = np.full((n, 2, 64), -1, np.int16)
    for i in np.flatnonzero(~terminal):
        c[i, 0, 0] = i % 640
        c[i, 1, 0] = 10240 + i % 640
    return ExperimentalCorpus(
        a,
        b,
        c,
        targets,
        terminal,
        offsets,
        np.array(["train"] * 9 + ["early_stop"] * 2 + ["screen"]),
        np.concatenate([np.arange(size) for size in sizes]).astype(np.int64),
    )


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def run(args):
    torch.set_num_threads(1)
    if args.interrupt:
        save = Runtime.checkpoint

        def interrupted(self, model, optimizer, signature, progress):
            save(self, model, optimizer, signature, progress)
            if progress["epoch"] == 1:
                os._exit(42)

        Runtime.checkpoint = interrupted
    rank = int(os.environ.get("RANK", "0"))
    arms = (args.arm,) if args.arm else ("A", "B", "C")
    for arm in arms:
        base = args.destination / arm
        folder = base / ("full" if args.mode == "full" else "resumed")
        options = RunOptions(
            device=args.device,
            checkpoint=folder / "checkpoint.pt",
            resume=folder / "checkpoint.pt" if args.mode == "resume" else None,
            stop_after_epochs=1 if args.mode == "segment" else None,
        )
        try:
            report, model = train(
                fixture(),
                arm,
                Hyperparameters(seed=11, max_epochs=3, patience=3, batch_groups=4),
                options=options,
                input_identity="synthetic-arithmetic",
            )
            atomic_save(
                {"model": model.state_dict(), "report": report}, folder / f"result-rank-{rank}.pt"
            )
        except TrainingPaused:
            if args.mode != "segment":
                raise
        if args.mode == "resume":
            for name in ("checkpoint.pt", f"result-rank-{rank}.pt"):
                left = torch.load(base / "full" / name, weights_only=True, map_location="cpu")
                right = torch.load(base / "resumed" / name, weights_only=True, map_location="cpu")
                assert equal(left, right), f"{arm}: interrupted continuation differs"
        if rank == 0:
            print(f"{arm} {args.mode}: synthetic recovery check passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("mode", choices=("full", "segment", "resume"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--arm", choices=("A", "B", "C"))
    parser.add_argument("--interrupt", action="store_true")
    run(parser.parse_args())
