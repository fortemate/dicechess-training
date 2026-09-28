"""Single-node CPU/CUDA execution and atomic, epoch-boundary training checkpoints.

A checkpoint is a private training artifact, distinct from exported inference weights. It is
complete only after every rank has contributed its random state and rank zero has replaced the
previous file. Platform output persistence is a separate responsibility (see docs/gpu-training.md).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import random
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class TrainingRuntimeError(ValueError):
    """An invalid runtime request, with messages that do not expose private paths."""


class TrainingPaused(Exception):
    """The requested segment finished and its checkpoint was saved; training is incomplete."""


@dataclass(frozen=True)
class RunOptions:
    device: str = "auto"
    checkpoint: Path | None = None
    resume: Path | None = None
    stop_after_epochs: int | None = None

    def __post_init__(self):
        if self.device not in ("auto", "cpu", "cuda"):
            raise TrainingRuntimeError("device must be auto, cpu or cuda")
        if self.stop_after_epochs is not None and (
            self.stop_after_epochs < 1 or self.checkpoint is None
        ):
            raise TrainingRuntimeError(
                "a bounded run needs a checkpoint and a positive epoch limit"
            )


def primary() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def identity(config: dict, *arrays: np.ndarray) -> str:
    """Hash effective inputs, including their shape/type, without copying an entire corpus."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str((contiguous.shape, contiguous.dtype.str)).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def atomic_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        # Flush the directory entry as well as the checkpoint's contents on POSIX hosts.
        if os.name == "posix":
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _random_state(device: torch.device) -> dict:
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
    }


def _restore_random(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))


_GROUP_SEQUENCE = itertools.count()


class Runtime:
    """A process per device; global batches are partitioned without padding or dropping data."""

    def __init__(self, options: RunOptions):
        self.options = options
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise TrainingRuntimeError("invalid distributed rank configuration")
        kind = options.device
        if kind == "auto":
            kind = "cuda" if torch.cuda.is_available() else "cpu"
        if kind == "cuda" and not torch.cuda.is_available():
            raise TrainingRuntimeError("CUDA was requested but is unavailable")
        if kind == "cuda" and not 0 <= local_rank < torch.cuda.device_count():
            raise TrainingRuntimeError("local rank has no CUDA device")
        self.device = torch.device(kind, local_rank) if kind == "cuda" else torch.device("cpu")
        self.owns_group = False

    def __enter__(self):
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if self.world_size > 1:
            if dist.is_initialized():
                raise TrainingRuntimeError("training requires its own process group")
            timeout = timedelta(minutes=5)
            store, rank, world_size = next(dist.rendezvous("env://", timeout=timeout))
            # The torchrun agent's store outlives a trainer. Isolate repeated runs/seeds so a
            # fresh group cannot consume stale rendezvous keys from the preceding trainer.
            store = dist.PrefixStore(f"training-{next(_GROUP_SEQUENCE)}", store)
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo",
                store=store,
                rank=rank,
                world_size=world_size,
                timeout=timeout,
            )
            self.owns_group = True
        return self

    def __exit__(self, *_):
        if self.owns_group:
            dist.destroy_process_group()
        torch.use_deterministic_algorithms(
            self.previous_deterministic, warn_only=self.previous_warn_only
        )

    def seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)

    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        model.to(self.device)
        if self.world_size == 1:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    def partition(self, batch):
        """Keep complete groups/rows; an empty rank still participates via a zero-weight loss."""
        local = batch[self.rank :: self.world_size]
        weight = len(local) * self.world_size / len(batch)
        return (local if len(local) else batch[:1]), weight

    def mean(self, value: float) -> float:
        if self.world_size == 1:
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor)
        return tensor.item() / self.world_size

    def environment(self) -> dict:
        root = Path(__file__).parent
        source = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            source.update(path.relative_to(root).as_posix().encode())
            source.update(path.read_bytes())
        return {
            "source_sha256": source.hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_threads": torch.get_num_threads(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "world_size": self.world_size,
            "device": self.device.type,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
            ),
        }

    def restore(self, model, optimizer, signature: str) -> dict | None:
        if self.options.resume is None:
            return None
        payload = torch.load(self.options.resume, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != "training-epoch-v1"
            or payload.get("identity") != signature
            or payload.get("environment") != self.environment()
        ):
            raise TrainingRuntimeError("checkpoint inputs, configuration or runtime do not match")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        _restore_random(payload["random"][self.rank], self.device)
        return payload["progress"]

    def checkpoint(self, model, optimizer, signature: str, progress: dict) -> None:
        if self.options.checkpoint is None:
            return
        state = _random_state(self.device)
        states = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(states, state)
        else:
            states[0] = state
        failed = torch.zeros((), dtype=torch.int32, device=self.device)
        if self.rank == 0:
            try:
                atomic_save(
                    {
                        "schema": "training-epoch-v1",
                        "identity": signature,
                        "environment": self.environment(),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "random": states,
                        "progress": progress,
                        # Both trainers use a constant learning rate, with no scheduler.
                        "scheduler": None,
                    },
                    self.options.checkpoint,
                )
            except OSError:
                failed.fill_(1)
        if self.world_size > 1:
            dist.broadcast(failed, src=0)
        if failed.item():
            raise TrainingRuntimeError("checkpoint save failed; verify output before resuming")

    def stop(self, epochs_this_run: int, complete: bool) -> None:
        limit = self.options.stop_after_epochs
        if not complete and limit is not None and epochs_this_run >= limit:
            raise TrainingPaused("segment saved; resume from its checkpoint to finish training")
