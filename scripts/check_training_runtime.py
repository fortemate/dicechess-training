"""Exercise the actual trainers on arithmetic fixtures, with no engine or private corpus.

Run this script in full/segment/resume modes with the same destination and launch topology.
The resume command compares full checkpoints and exported models against the uninterrupted run.
Use --interrupt to terminate workers immediately after their first complete epoch checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dicechess_training.prerank.dataset import Corpus
from dicechess_training.prerank.train import Hyperparameters, train
from dicechess_training.runtime import RunOptions, Runtime, TrainingPaused, atomic_save, primary
from dicechess_training.train import train_value_model


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def fixture() -> Corpus:
    """Uneven group sizes and an uneven last global batch, including a rank with no groups."""
    sizes = np.array([3, 7, 5, 4, 6, 3, 8, 4, 5, 6, 7])
    offsets = np.concatenate(([0], sizes.cumsum()))
    features = np.sin(np.arange(int(offsets[-1]) * 9).reshape(-1, 9) / 13).astype(np.float32)
    return Corpus(
        features=features,
        targets=(features[:, 0] - features[:, 3] * 2).astype(np.float64),
        offsets=offsets,
        splits=np.array(["train"] * 9 + ["validation"] * 2),
        columns=("material_diff",) + tuple(f"synthetic_{i}" for i in range(1, 9)),
        manifest={
            "engine_version": "synthetic-none",
            "teacher": {"id": "arithmetic"},
            "source_sha256": "0" * 64,
            "groups_sha256": "1" * 64,
        },
    )


def run(args):
    torch.set_num_threads(1)
    if args.interrupt:
        save = Runtime.checkpoint

        def interrupted(self, *values):
            save(self, *values)
            os._exit(42)

        Runtime.checkpoint = interrupted
    rank = int(os.environ.get("RANK", "0"))
    names = (args.trainer,) if args.trainer else ("value", "prerank")
    for name in names:
        path = args.destination / name
        folder = path / ("full" if args.mode == "full" else "resumed")
        options = RunOptions(
            device=args.device,
            checkpoint=folder / "checkpoint.pt",
            resume=(folder / "checkpoint.pt") if args.mode == "resume" else None,
            stop_after_epochs=1 if args.mode == "segment" else None,
        )
        try:
            if name == "value":
                x = np.sin(np.arange(9 * 774).reshape(9, 774) / 23).astype(np.float32)
                y = ((x[:, 0] + 1) / 2).astype(np.float32)
                model = train_value_model(x, y, epochs=3, batch_size=4, hidden=8, options=options)
                result = {"model": model.state_dict()}
            else:
                report, model, scores = train(
                    fixture(),
                    Hyperparameters(
                        hidden_dims=(8,), max_epochs=3, patience=3, batch_groups=4, k=2
                    ),
                    options=options,
                )
                result = {
                    "model": model.state_dict(),
                    "report": report,
                    "scores": torch.from_numpy(scores),
                }
            atomic_save(result, folder / f"result-rank-{rank}.pt")
        except TrainingPaused:
            if args.mode != "segment":
                raise
        if args.mode == "resume":
            for filename in ("checkpoint.pt", f"result-rank-{rank}.pt"):
                left = torch.load(path / "full" / filename, weights_only=True, map_location="cpu")
                right = torch.load(
                    path / "resumed" / filename, weights_only=True, map_location="cpu"
                )
                assert equal(left, right), f"{name}: resumed {filename} differs"
        if primary():
            print(
                json.dumps(
                    {
                        "trainer": name,
                        "mode": args.mode,
                        "complete": args.mode != "segment",
                        "resume_equal": args.mode == "resume",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("mode", choices=("full", "segment", "resume"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--trainer", choices=("value", "prerank"))
    parser.add_argument("--interrupt", action="store_true")
    run(parser.parse_args())
