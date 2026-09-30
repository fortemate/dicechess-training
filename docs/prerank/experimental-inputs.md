# Experimental full-state pre-ranker inputs

This offline contract is separate from `playground-prerank-groups-v1` and the serving
pre-ranker manifest. It prepares comparable representations, not a serving artifact or a
model-quality conclusion. The private **Safety-aware pre-ranker pilot** protocol supplies
cohort admission, frozen splits, teacher identity, budgets and later evaluation gates.

`tools/prerank-experimental` pins engine 0.14.0. It reads six-field root and afterstate
FENs from a previously replay-certified corpus. It checks root-mover perspective, rich9
parity and exact terminal sentinel agreement. It does not certify stored paths, list
completeness, nonterminal teacher targets or data rights: those are upstream admission gates.
The source file digest binds its output to that admission evidence. All output stays with
the private corpus.

Run the extractor with a **new** output directory:

```sh
cd tools/prerank-experimental
sbt test
sbt 'runMain ExperimentalFeatures <groups.json> <new-feature-directory>'
```

A completed export has `features.jsonl` and `manifest.json`; without the manifest a
partial export is unusable. The extractor refuses existing output directories and checks
for trailing input and for input changes. There is no normalized-FEN grouping or new sampling.

The three versioned representation definitions are:

- **A / exp-rich9-root-v1:** engine rich9 afterstate features from the original mover's perspective.
- **B / exp-rich9-safety-root-v1:** A followed by own attacked, own attacked-and-undefended,
  opponent attacked, opponent attacked-and-undefended non-king counts. `allAttackers`
  answers boolean attacks/defence; pawn diagonals, sliders, kings and knights use engine rules.
- **C / exp-king-relative-root-v1:** own and opponent king anchors with 4x4 buckets;
  non-king categories are own P/N/B/R/Q then opponent P/N/B/R/Q. Black movers receive a
  vertical reflection (`square XOR 56`). Bucket = file/2 + 4*(rank/2), with zero-based coordinates.
  ID = ((anchor*16 + bucket)*10 + category)*64 + square, in separate anchor halves.
  IDs are sorted, and each anchor is padded with -1 to 64 slots. Padding contributes nothing.

`prerank.experimental.pack` requires independently pinned source and frozen split-map
SHA256 values. It keeps complete candidate lists and row order, checks all assignments,
rejects output reuse and writes numeric `arrays.npz` plus a completion manifest. It stores
only a within-group canonical move-order permutation for prediction tie-breaking; raw
paths and game IDs are not copied into arrays. Source-game clustering and overlap evidence
must remain bound separately before evaluation. Existing split assignment code is not used.
Only `train`, `early_stop` and `screen` are supported: this contract cannot relabel development
data as acceptance.

```python
from pathlib import Path
from dicechess_training.prerank.experimental import pack, load, digest, make_model

report = pack(
    Path("<features>"),
    Path("<frozen-group-splits.json>"),
    Path("<new-arrays>"),
    source_sha256="<admitted-source-sha256>",
    splits_sha256="<frozen-splits-sha256>",
)
corpus = load(Path("<new-arrays>"), manifest_sha256="<pinned-completion-sha256>")
model = make_model(corpus, "A")  # also B or C
```

A and B use a 32/32 ReLU MLP with train-only nonterminal mean/std, zero variance scale=1.
C sums 4-D embeddings for each anchor and concatenates eight values into a 32/32 ReLU MLP.
It is stateless; this does not implement incremental NNUE. Capacity differs and parameter
counts must be reported. Dense statistics exclude terminal rows because terminal inference
bypasses every model. This refinement must be recorded in the private run protocol before fits.

Terminal candidates retain their exact target and group membership. C has no anchor IDs for
them and never invents a missing king. `score_candidates` evaluates only nonterminal rows and
assigns terminal rows positive infinity **for ordering only**. Do not feed this inference helper
into ListNet. The recoverable runner below defines terminal-group loss handling and early-stop
selection separately. Cloud budgets and the private run plan must be frozen before real fits.
No cloud transfer, serving export, strength or latency qualification follows from input preparation.

Synthetic verification covers pawn pushes versus attacks, defended pieces, both colors,
exact bucket/category/square IDs, absent terminal anchors, model gradients, train-only scaling,
padding, pinned digest failures, complete list boundaries and terminal bypass. JVM tests are a
separate gate from `mise run check`.

## Recoverable experimental training

`prerank.experimental_train` trains one arm and seed with the same complete lists, target gains,
NumPy shuffle sequence and global group batches. Architecture initialization uses a separate
Torch RNG. Every epoch records the group-order digest and optimizer-update count. Dense scaling
continues to use train nonterminal rows only. Early stopping selects the lowest **early-stop
ListNet loss**, not recall or a screen result. No screen predictions or metrics are produced.

For a group's full inverse-square target distribution `p`, let `t` be terminal mass and
`m = 1-t` nonterminal mass. The fixed terminal branch assigns each terminal its target gain;
the learned branch assigns `m * softmax(score)` to the remaining candidates. Cross entropy is
therefore `-sum(nonterminal p * log_softmax(score))` plus the fixed terminal entropy and
`-m*log(m)`. It retains the original full-list gains and group denominator. A wholly terminal
group contributes its constant entropy and zero model gradient. All arms use this same loss.
Inference retains the exact terminal-first ordering; its infinities never enter training.

A single-node `Runtime` handles CPU, CUDA and `torchrun` DDP. Rank partitioning keeps groups
whole and weights each rank by its share of the global batch. Empty ranks participate with a
zero-weight forward on a valid nonterminal row. Uneven final batches are not dropped or padded
into the objective. Synthetic tests compare the global objective/gradients with uneven rank
partitions and compare group-relative predictions; FP32 optimizer bias roundoff can differ
between serial and DDP. Recovery with the **same** topology is tested exactly for every weight,
optimizer state, history and RNG state.

Freeze a private plan before fits. Example structure (substitute the admitted input hash and
protocol-approved settings):

```json
{
  "schema": "experimental-prerank-fit-v1",
  "input_manifest_sha256": "<pinned-input-manifest-sha256>",
  "arm": "A",
  "hyperparameters": {"seed": 11, "learning_rate": 0.001, "batch_groups": 64,
                      "max_epochs": 40, "patience": 5}
}
```

Use identical hyperparameters for A/B/C for each seed. Record C's different capacity. The CLI
requires both the input-manifest hash and plan hash. Outputs must lie in the current working
directory or temporary directory. A fresh run refuses an existing destination:

```sh
python -m dicechess_training.prerank.experimental_train \
  <arrays-directory> <plan.json> <new-run-directory> \
  --manifest-sha256 <input-hash> --plan-sha256 <plan-hash> --device cpu \
  --stop-after-epochs 1
# Exit 75 means a paused segment, not completion. Continue with the same plan/runtime:
python -m dicechess_training.prerank.experimental_train \
  <arrays-directory> <plan.json> <same-run-directory> \
  --manifest-sha256 <input-hash> --plan-sha256 <plan-hash> --device cpu --resume
```

Use `torchrun --nproc_per_node=<devices> -m dicechess_training.prerank.experimental_train ...`
for DDP. Every rank must use the same plan and path arguments. The CLI writes `checkpoint.pt`
initially and at every complete epoch; `result.pt` appears only after epoch cap/early stopping
completes. The checkpoint keeps current training weights/optimizer, selected best weights,
early-stop history/counter, shuffle state and every rank's Python/NumPy/Torch/device RNG state.
Learning rate is constant, with no scheduler. Checkpoints use atomic replacement and safe
`weights_only` loading through the existing runtime. Source, software, thread count, device,
world size, plan and input identities must match on resume. Resuming a completed checkpoint
makes no further optimizer update. A mid-epoch interruption returns to its last saved boundary;
platform persistence and durable download remain operator responsibilities.

`scripts/check_experimental_runtime.py` checks uninterrupted versus segmented continuation for
all arms in fresh processes. Its `--interrupt` option terminates the actual worker immediately
after the first complete epoch checkpoint. CPU and two-process CPU recovery are repository
gates; CUDA and two-GPU recovery must be rerun on the actual cloud hardware before real fits.
This runner does not authorize a cloud transfer, enforce account-wide GPU quotas or provide the
independent tactical/acceptance/latency gates of the private pilot.
