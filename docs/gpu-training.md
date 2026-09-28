# CPU, CUDA and resumable training

The existing value trainer (`train_value_model`) and listwise pre-ranker (`prerank train`)
select CUDA when available and CPU otherwise. `--device cpu` or `RunOptions(device="cpu")`
keeps local work on CPU; requesting CUDA explicitly fails if it is unavailable. Features and
teacher labels still come from the existing engine pipeline. This runtime does not generate them.

## Environment

Ordinary `uv sync --locked` retains the CPU-only Linux dependency group used by CI. CUDA is an
explicit, mutually exclusive option:

```sh
uv sync --locked --no-group cpu --extra cu128
uv run --no-group cpu --extra cu128 python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

Use the same options on subsequent `uv run` commands, or run `.venv/bin/python` directly.
A plain `uv run` reselects the default CPU group. The extra targets Linux CUDA 12.8; it does not
install a driver. The source selection follows [uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).

Kaggle already supplies CUDA-enabled PyTorch. To use that pinned environment without replacing it,
attach a reviewed source snapshot (including the committed golden fixtures) and set `PYTHONPATH`
to its `src` directory. Record the source revision and environment versions. Do not run ordinary
`uv sync` over the notebook's preinstalled environment. Alternatively, install the CUDA dependency
set above in a separate environment when the notebook has approved package/network access.

## Launch

A single-device run uses the existing command:

```sh
python -m dicechess_training.prerank train <corpus-dir> <runs-dir> --device cuda
```

For two GPUs, launch one process per device. `batch_groups` remains a **global** batch size.
Complete groups are divided across ranks without duplicating or dropping a short final batch;
an empty rank participates with zero contribution. All ranks must see the same admitted input.
Validation reads the same validation split on each rank. Final reports and inference weights
are written by rank zero only.

```sh
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dicechess_training.prerank train <corpus-dir> <runs-dir> --device cuda
```

The value API uses the same runtime. In a script launched by Python or torchrun:

```python
from pathlib import Path
from dicechess_training.runtime import RunOptions
from dicechess_training.train import train_value_model

model = train_value_model(
    train_x,
    train_y,
    options=RunOptions(device="auto", checkpoint=Path("value-checkpoint.pt")),
)
```

Returned models are on CPU for the existing evaluation/export callers. The pre-ranker retains
float64 training and its existing loss and early-stop rules. The value model retains float32.
DDP reductions may differ numerically from serial execution; compare experiments under the same
launch topology rather than treating hardware changes as exact replications.

## Checkpoint and resume

After each completed epoch, the runtime atomically replaces a full checkpoint. It records model,
Adam, every rank's Python/NumPy/PyTorch CPU/CUDA random state, epoch progress, and (for the
pre-ranker) shuffle-generator state, history, early stopping and best weights. Both trainers use
a constant learning rate, so the scheduler field is explicitly `None`. Model-only `weights-*.pt`
remain the export format and cannot resume training.

Resume requires identical inputs, configuration, source files, process count, device type and
recorded runtime versions/settings. Mismatches fail closed. Checkpoints use
`torch.load(..., weights_only=True)`; only restore checkpoints you control. Keep them private.

For a bounded Kaggle segment, choose exactly one protocol seed and an epoch count that fits well
inside the remaining session time. Substitute an allowed protocol seed for `<seed>`:

```sh
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dicechess_training.prerank train <corpus-dir> <working-runs-dir> \
  --device cuda --seeds <seed> --stop-after-epochs <segment-epochs>
```

A paused segment exits zero with `PAUSED`, writes `checkpoint-seed-<seed>.pt`, and does not
create final inference weights or a completed report. It is not a completed experiment. Use a
fresh output directory for each segment so old final outputs cannot be mistaken for new ones.
After Kaggle's **Save & Run All** version finishes successfully, verify the checkpoint exists in
that version's output. Attach that exact version as input to a new private notebook, then run:

```sh
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dicechess_training.prerank train <same-corpus-dir> <new-working-runs-dir> \
  --device cuda --seeds <seed> --resume <committed-checkpoint.pt> \
  --stop-after-epochs <segment-epochs>
```

Omit the epoch limit to finish. A completed pre-ranker still exits 2 when it fails its existing
admissibility floor; torchrun reports that nonzero result as a worker failure. Inspect the written
report rather than interpreting every launcher failure as a missing checkpoint.

**Durability boundary:** an abrupt worker failure can lose the current epoch. An abrupt VM loss
can lose every file that exists only in `/kaggle/working`; atomic saving is not external storage.
Recovery is guaranteed only from a checkpoint already present in a committed output or another
approved durable destination. Keep the previous committed version until its successor is verified.
If directory synchronization fails after replacement, the local destination may already contain
the new checkpoint. The runtime reports failure; treat that output as unverified and recover from
the preceding committed version. The atomic writer does not certify storage durability after a
failed filesystem synchronization.

## Mechanical checks and data admission

`scripts/check_training_runtime.py` exercises both actual trainers with arithmetic arrays,
uneven batches and exact fresh-process recovery. It does not imitate engine features or establish
model quality. Run `full`, then `segment`, then `resume` against one scratch directory, using the
same launch command for every stage:

```sh
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/check_training_runtime.py <scratch-dir> full --device cuda
# Repeat with segment, then resume. Resume asserts equality of the full training state.
```

For an abrupt-worker test, use `segment --trainer value --interrupt` (expected exit 42 from the
worker), then `resume --trainer value`; repeat for `prerank`. To test cross-session recovery, commit
the full and interrupted checkpoint outputs first, then copy them into the next session's writable
scratch directory and run only `resume`. The CPU test suite also exercises Gloo with two processes.

Private notebook visibility alone does not approve a corpus upload. Before real-data use, resolve
source licensing, platform terms, full-state/feature provenance and independent evaluation gates.
The runtime smoke uses synthetic arrays only and cannot settle those requirements.
