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
into ListNet: the training runner must freeze terminal-group loss handling, early-stop loss,
identical batching, checkpoint/recovery and device budgets before real model comparison.
No real training runner is added here. No cloud transfer, serving export, strength or latency
qualification follows from successful input preparation.

Synthetic verification covers pawn pushes versus attacks, defended pieces, both colors,
exact bucket/category/square IDs, absent terminal anchors, model gradients, train-only scaling,
padding, pinned digest failures, complete list boundaries and terminal bypass. JVM tests are a
separate gate from `mise run check`.
