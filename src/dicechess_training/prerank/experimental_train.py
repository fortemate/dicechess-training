"""Development-only A/B/C training with group-mean ListNet and epoch recovery.

Terminal ordering is exact. For training, the terminal branch has its target probability mass;
the learned branch is a conditional softmax over nonterminal candidates. Full-list target gains
and all groups are retained. Never pass inference infinity sentinels into the loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dicechess_training.prerank.dataset import target_gains
from dicechess_training.prerank.experimental import (
    ExperimentalCorpus,
    load,
    make_model,
    require,
)
from dicechess_training.prerank.model import listwise_loss
from dicechess_training.runtime import (
    RunOptions,
    Runtime,
    TrainingPaused,
    atomic_save,
    identity,
    primary,
)

PROTOCOL = "experimental-prerank-fit-v1"
CHECKPOINT_FILE = "checkpoint.pt"
RESULT_FILE = "result.pt"


@dataclass(frozen=True)
class Hyperparameters:
    seed: int = 11
    learning_rate: float = 0.001
    batch_groups: int = 64
    max_epochs: int = 40
    patience: int = 5

    def __post_init__(self):
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "invalid seed")
        require(
            type(self.learning_rate) in (float, int)
            and math.isfinite(self.learning_rate)
            and self.learning_rate > 0,
            "invalid learning rate",
        )
        for value in (self.batch_groups, self.max_epochs, self.patience):
            require(type(value) is int and value > 0, "training limits must be positive integers")


DEFAULTS = Hyperparameters()


@dataclass(frozen=True)
class Prepared:
    gains: np.ndarray
    constants: np.ndarray
    fitting: np.ndarray
    measuring: np.ndarray
    dummy_row: int


def prepare(corpus: ExperimentalCorpus) -> Prepared:
    """Targets are ranked inside complete lists; terminal mass is not renormalized away."""
    corpus.validate()
    gains = np.zeros(len(corpus.targets), np.float32)
    constants = np.zeros(corpus.groups, np.float64)
    nonterminal_counts = np.zeros(corpus.groups, np.int64)
    for group in np.concatenate((corpus.of("train"), corpus.of("early_stop"))):
        lo, hi = corpus.offsets[group : group + 2]
        distribution = target_gains(corpus.targets[lo:hi])
        terminal = corpus.terminal[lo:hi]
        gains[lo:hi] = distribution
        nonterminal_counts[group] = (~terminal).sum()
        fixed = distribution[terminal]
        constants[group] = -np.sum(fixed * np.log(fixed))
        mass = distribution[~terminal].sum()
        if mass > 0:
            constants[group] -= mass * np.log(mass)
    sizes = np.diff(corpus.offsets)
    fitting = corpus.of("train")
    fitting = fitting[sizes[fitting] >= 2]
    measuring = corpus.of("early_stop")
    measuring = measuring[sizes[measuring] >= 2]
    require(np.any(nonterminal_counts[fitting] >= 2), "train has no learned ordering")
    require(np.any(nonterminal_counts[measuring] >= 2), "early-stop has no learned ordering")
    train_rows = np.concatenate(
        [np.arange(corpus.offsets[g], corpus.offsets[g + 1]) for g in fitting]
    )
    dummy = int(train_rows[~corpus.terminal[train_rows]][0])
    return Prepared(gains, constants, fitting, measuring, dummy)


def batch_loss(
    model: nn.Module,
    corpus: ExperimentalCorpus,
    prepared: Prepared,
    groups: np.ndarray,
    arm: str,
    device: torch.device,
) -> torch.Tensor:
    """Full-list cross entropy with an exact terminal branch and learned conditional branch."""
    rows = np.concatenate([np.arange(corpus.offsets[g], corpus.offsets[g + 1]) for g in groups])
    index = np.repeat(np.arange(len(groups)), np.diff(corpus.offsets)[groups])
    nonterminal = ~corpus.terminal[rows]
    features = {"A": corpus.a, "B": corpus.b, "C": corpus.c}[arm]
    constant = torch.tensor(prepared.constants[groups].mean(), dtype=torch.float32, device=device)
    if not nonterminal.any():
        # All ranks must call the wrapped forward, including ranks containing only exact rows.
        dummy = torch.from_numpy(features[prepared.dummy_row : prepared.dummy_row + 1]).to(device)
        return model(dummy).sum() * 0 + constant
    inputs = torch.from_numpy(features[rows[nonterminal]]).to(device)
    weights = torch.from_numpy(prepared.gains[rows[nonterminal]]).to(device)
    group_index = torch.from_numpy(index[nonterminal]).to(device)
    return listwise_loss(model(inputs), weights, group_index, len(groups)) + constant


def validation_loss(model, corpus, prepared, arm, batch_groups, runtime) -> float:
    """Group mean on early-stop only, without scoring or selecting on screen."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(prepared.measuring), batch_groups):
            global_batch = prepared.measuring[start : start + batch_groups]
            local, weight = runtime.partition(global_batch)
            loss = batch_loss(model, corpus, prepared, local, arm, runtime.device)
            total += runtime.mean(float(loss) * weight) * len(global_batch)
    return total / len(prepared.measuring)


def train(
    corpus: ExperimentalCorpus,
    arm: str,
    hyper: Hyperparameters = DEFAULTS,
    *,
    options: RunOptions | None = None,
    input_identity: str = "",
) -> tuple[dict, nn.Module]:
    """One arm/seed. The caller pins its run plan and manages private artifact persistence."""
    require(arm in ("A", "B", "C"), "unknown experimental arm")
    prepared = prepare(corpus)
    with Runtime(options or RunOptions()) as runtime:
        runtime.seed(hyper.seed)
        rng = np.random.default_rng(hyper.seed)  # independent of architecture initialization
        model = make_model(corpus, arm)
        require(
            all(p.dtype == torch.float32 for p in model.parameters()), "float32 parameters required"
        )
        wrapped = runtime.wrap(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper.learning_rate, weight_decay=0.0)
        signature = identity(
            {"protocol": PROTOCOL, "arm": arm, "hyper": asdict(hyper), "input": input_identity},
            corpus.a,
            corpus.b,
            corpus.c,
            corpus.targets,
            corpus.terminal,
            corpus.offsets,
            corpus.splits,
            corpus.tie_order,
        )
        progress = {
            "epoch": 0,
            "complete": False,
            "history": [],
            "best_loss": math.inf,
            "best_epoch": 0,
            "best_state": None,
            "shuffle_rng": json.dumps(rng.bit_generator.state),
        }
        restored = runtime.restore(model, optimizer, signature)
        if restored is not None:
            progress = restored
            rng.bit_generator.state = json.loads(progress["shuffle_rng"])
        else:
            runtime.checkpoint(model, optimizer, signature, progress)
        start_epoch = progress["epoch"]
        for epoch in range(start_epoch + 1, hyper.max_epochs + 1):
            if progress["complete"]:
                break
            train_loss, order_hash, updates = _epoch(
                wrapped, optimizer, corpus, prepared, arm, hyper, rng, runtime
            )
            valid = validation_loss(model, corpus, prepared, arm, hyper.batch_groups, runtime)
            require(math.isfinite(train_loss) and math.isfinite(valid), "non-finite training loss")
            progress["history"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "early_stop_loss": valid,
                    "order_sha256": order_hash,
                    "updates": updates,
                }
            )
            if valid < progress["best_loss"]:
                progress.update(
                    best_loss=valid,
                    best_epoch=epoch,
                    best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
            progress.update(
                epoch=epoch,
                complete=(
                    epoch == hyper.max_epochs or epoch - progress["best_epoch"] >= hyper.patience
                ),
                shuffle_rng=json.dumps(rng.bit_generator.state),
            )
            runtime.checkpoint(model, optimizer, signature, progress)
            runtime.stop(epoch - start_epoch, progress["complete"])
        model.load_state_dict(progress["best_state"])
        report = {
            "protocol": PROTOCOL,
            "arm": arm,
            "hyperparameters": asdict(hyper),
            "identity": signature,
            "input_identity": input_identity,
            "complete": progress["complete"],
            "history": progress["history"],
            "best_epoch": progress["best_epoch"],
            "best_loss": progress["best_loss"],
            "train_groups": len(prepared.fitting),
            "early_stop_groups": len(prepared.measuring),
            "parameters": sum(p.numel() for p in model.parameters()),
            "environment": runtime.environment(),
            "role": "development fit; no qualification",
        }
        return report, model.cpu().eval()


def _epoch(model, optimizer, corpus, prepared, arm, hyper, rng, runtime):
    model.train()
    order = rng.permutation(prepared.fitting)
    total = 0.0
    for start in range(0, len(order), hyper.batch_groups):
        global_batch = order[start : start + hyper.batch_groups]
        local, weight = runtime.partition(global_batch)
        optimizer.zero_grad()
        loss = batch_loss(model, corpus, prepared, local, arm, runtime.device) * weight
        require(torch.isfinite(loss), "non-finite batch loss")
        loss.backward()
        optimizer.step()
        total += runtime.mean(float(loss.detach())) * len(global_batch)
    return (
        total / len(order),
        hashlib.sha256(memoryview(order).cast("B")).hexdigest(),
        math.ceil(len(order) / hyper.batch_groups),
    )


def _safe_write_path(path: Path) -> Path:
    resolved = path.resolve()
    require(
        resolved.is_relative_to(Path.cwd().resolve())
        or resolved.is_relative_to(Path(tempfile.gettempdir()).resolve()),
        "output must be inside the working directory or temporary directory",
    )
    return resolved


def _run(args) -> int:
    corpus = load(args.corpus.resolve(), manifest_sha256=args.manifest_sha256)
    plan_bytes = args.plan.resolve().read_bytes()
    require(hashlib.sha256(plan_bytes).hexdigest() == args.plan_sha256, "plan digest differs")
    plan = json.loads(plan_bytes)
    require(plan["schema"] == PROTOCOL, "unsupported training plan")
    require(plan["input_manifest_sha256"] == args.manifest_sha256, "plan input differs")
    hyper = Hyperparameters(**plan["hyperparameters"])
    require(plan["arm"] in ("A", "B", "C"), "unknown experimental arm")
    destination = _safe_write_path(args.destination)
    options = RunOptions(
        device=args.device,
        checkpoint=destination / CHECKPOINT_FILE,
        resume=destination / CHECKPOINT_FILE if args.resume else None,
        stop_after_epochs=args.stop_after_epochs,
    )
    if args.resume:
        require((destination / CHECKPOINT_FILE).is_file(), "resume checkpoint is missing")
    elif primary():
        destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report, model = train(
        corpus,
        plan["arm"],
        hyper,
        options=options,
        input_identity=args.manifest_sha256 + ":" + args.plan_sha256,
    )
    if primary():
        atomic_save({"report": report, "model": model.state_dict()}, destination / RESULT_FILE)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "elapsed_seconds": time.monotonic() - started,
                    "identity": report["identity"],
                }
            )
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int)
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except TrainingPaused:
        if primary():
            print(json.dumps({"status": "paused", "checkpoint": "epoch boundary"}))
        return 75
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, pickle.UnpicklingError):
        if primary():
            print("refused: invalid inputs, runtime or artifact destination", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
