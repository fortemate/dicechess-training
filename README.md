# dicechess-training

[![CI](https://github.com/fortemate/dicechess-training/actions/workflows/ci.yml/badge.svg)](https://github.com/fortemate/dicechess-training/actions/workflows/ci.yml)

Open training pipeline for the [Fortemate](https://fortemate.com) Dice Chess project — a GPU
**label factory** that manufactures deep exact-search labels, and two small networks trained on
them, designed to scale from a single workstation to an HPC cluster.

> [!NOTE]
> Fortemate is an independent project and is not affiliated with or endorsed by
> [dicechess.com](https://dicechess.com). We use a compatible Dice Chess ruleset and observe
> publicly visible games for research; links to that service are references, not claims of
> ownership or partnership.

> **Status: active development.** Pipeline roadmap:
>
> - [ ] Transposition table + enabling depth-3 search on the engine's already-landed
>       Star1/Star2 pruning, gated by a depth-3 vs depth-2 arena experiment
> - [x] First PyTorch → ONNX training stack, validated end-to-end at toy scale
>       ([#3](https://github.com/fortemate/dicechess-training/issues/3))
> - [x] Training data schema v0: provenance-first Parquet shards, rawboard-768 + dice encoding
>       ([#2](https://github.com/fortemate/dicechess-training/issues/2))
> - [ ] Engine hooks: ONNX evaluator at chance nodes and a learned pre-ranker slot
> - [x] Public bot-vs-bot sample dataset collected from publicly observable games on an
>       independent platform
>       ([#4](https://github.com/fortemate/dicechess-training/issues/4))
> - [ ] CPU label-factory v0 ([#7](https://github.com/fortemate/dicechess-training/issues/7))
> - [ ] First real playground evaluation model ([#12](https://github.com/fortemate/dicechess-training/issues/12)):
>       contract mechanics in [ADR 0001](docs/decisions/0001-playground-train-serve-contract.md), feature schema
>       selected as S0 `kcp-13` by the predeclared ablation in [#17](https://github.com/fortemate/dicechess-training/issues/17)

## Getting started

Requires [mise](https://mise.jdx.dev/) (or plain [uv](https://docs.astral.sh/uv/)); everything
runs on CPU in well under a minute.

```bash
mise run setup   # install dependencies (uv sync)
mise run demo    # end-to-end: toy data → train → holdout metrics → ONNX export → parity check
mise run check   # lint + format + tests, mirrors CI
```

The demo trains on [`sample/playsite-bots-v0`](sample/README.md) — 49,000 positions from 2,999
publicly observable bot-vs-bot games recorded by Fortemate from an independent Dice Chess
service — and reports holdout log-loss against a no-information baseline, plus a calibration
table. Fortemate does not operate or represent the source service. Pass `--synthetic` to run
the same pipeline on random placements instead.

## Serving contracts

The private evaluation service behind the [protected playground](https://github.com/fortemate/dicechess-evaluation-playground)
serves one contract today: `standard-kcp` over the engine's 13-feature `kcp-13` schema — mover-perspective,
dice-free, output `P(side to move wins)`. [ADR 0001](docs/decisions/0001-playground-train-serve-contract.md)
records the inventory, keeps `kcp-13` as the reference baseline, and defers the first candidate's feature schema to a predeclared ablation.
`dicechess_training.contracts.kcp13` pins its layout, tensor names, manifest rules and perspective on the
Python side and fails closed on any mismatch; the feature values themselves are never reimplemented here —
`tests/fixtures/kcp13/` holds the golden vectors written by the engine through
[`tools/kcp13-golden`](tools/kcp13-golden/README.md) (`mise run golden:kcp13`, needs a JDK and sbt).

## Playground evaluation benchmark

The [versioned benchmark and promotion gate](docs/benchmark/README.md) evaluates the
ADR 0001 position contract with deterministic game-group splits, leakage audits, calibrated
binary scores and sealed final qualification. Recompute the public synthetic baseline with:

```bash
uv run python -m dicechess_training.benchmark --data tests/fixtures/benchmark
```

This fixture verifies benchmark mechanics; it provides no model-quality evidence. Real model
reports and final holdout data remain private.

### Retaining a bundle

A bundle is what the protocol binds and what a later reader has to be able to fetch, but it is
kept as a directory on whichever machine exported it. Publishing one to a dataset repository on
the [Hugging Face Hub](https://huggingface.co/docs/hub/) gives it a durable, revision-addressed
home:

```bash
uv tool install huggingface_hub   # provides `hf`; not a dependency of this project
hf auth login
uv run python -m dicechess_training.hub \
  --repo <owner>/<repository> --bundle <bundle-directory> --path-in-repo <run>/<bundle>
```

The bundle is admitted through the benchmark's own loader before anything is uploaded, and the
published copy is read back and compared by digest afterwards, so the result is a copy that is
proven equal to the export rather than assumed to be. Only the three files of the bundle contract
are published; anything else sitting beside them is left behind. The destination repository is
created private, because a bundle carries its own data terms. The command prints the revision it
produced — record it, since that is what makes a later read reproducible.

### Retaining a model package

A candidate package has the same problem and the same answer, with one difference in what can be
checked. Run this once a package is one you intend to keep — after it has been built on a commit
you can name, not while you are still iterating:

```bash
uv run python -m dicechess_training.hub \
  --repo <owner>/<repository> --package <package-directory> --path-in-repo <run>/<package>
```

A bundle is admitted by the benchmark's own loader. A package has no equivalent loader, but it is
self-describing: the manifest records what its own graph should hash to, so admission leans on
that claim rather than on a digest list kept beside it. Two checks, answering different questions
— the digest says the three files belong together, and the feature schema's ONNX contract says the
graph could actually be served, which a digest cannot tell you. A package whose declared schema
has no contract here is refused by name rather than passed through unchecked.

The destination is a **model** repository rather than a dataset one, and it is likewise created
private: trained weights are private wherever they are written, so that is not a flag.

## Why

Dice Chess is a chess variant where three piece-type dice are rolled each turn, and only the
piece types shown may move. Every turn therefore passes through a chance node with up to 216
outcomes (56 distinct multisets), and that explosion shapes everything we have measured while
building the engine, the platform, and the bots:

- **Search beats features.** Richer evaluation features gained +20 percentage points at
  1-ply search and nothing at 2-ply. Playing strength lives in depth.
- **Depth is capped at 2 plies** by the chance-node explosion. Star1/Star2 pruning has
  already landed in the engine and cut per-candidate roll work by ~68% at 2 plies; depth 3
  is the next step (transposition table first), and its price is what the label factory
  amortizes.
- **More data of the same quality no longer helps:** ≤1 pp per doubling of supervised
  training data from archived games, and LightGBM distillation gained nothing on 20M rows.
  Better labels are needed, not more labels.
- **A plain raw-board MLP is not enough.** Served as a 2-ply evaluator it measured −25 Elo
  against the feature-based model — so the value net's architecture sweep targets
  NNUE-style, king-relative representations rather than a bigger plain MLP.
- **Our strongest bot is not the ML one.** A hand-crafted evaluation with an exact 216-roll
  rescoring phase beats our best model-based bot 66.4% head-to-head — and under real-time
  budgets it only manages to exactly rescore 1–2 candidate moves. A learned move pre-ranker
  already measured **+4.8 pp** from improving which candidates get rescored.

The conclusion drives this repository: use HPC to manufacture **deep exact labels** that our
own hardware can never compute, and compress them into **small networks** that run at the
edge.

## The plan: a label factory and two nets

**Label factory** (the cluster-bound workload): CPU workers expand Star2-pruned depth-3
search trees over Fortemate-generated self-play positions and, where provenance and reuse
terms permit, positions from the research archive of publicly observable games. GPUs evaluate
leaves and exact 216-roll candidate rescores in large batches. One pass emits two training
signals — depth-3 value labels and exact per-candidate rescore distributions. Target: a
**100M+ position dataset** representing years of CPU search, computed in days.

Two small networks train on those labels:

1. **Chance-collapse value net** — an afterstate network that learns
   `V(after-move) ≈ E_dice[V(position, roll)]`, replacing the exact enumeration of a chance
   node with a single call. Small on purpose (it is called at every interior chance node);
   INT8 ONNX in the engine buys **depth 3–4 within the production time budget**. The key
   experiment: depth-3-with-net vs depth-2-exact at equal wall-clock.
2. **Listwise pre-ranker** — scores all legal turn paths so the right 1–2 candidates get the
   exact rescore. Trained against exact rescore distributions (a luxury deterministic chess
   does not have — no noisy MCTS visit counts), distilled to the phase-1 latency budget of
   CPU and edge bots. Its rank-hit metric and A/B arena protocol cannot fail to produce a
   number by the final presentation.

Evaluation combines holdout agreement with the depth-3 teacher (MSE, rank correlation,
log-loss/calibration) with head-to-head arena matches at fixed time controls, and ultimately
rated games on the public Glicko-2 bot ladder.

## Scaling and training goals

**Primary** — run the label factory at scale (target: 100M+ depth-3-labeled positions)
and train both networks on it, with labeling throughput and scaling measured and profiled.

**Distributed sweeps & optimization**

- Multi-GPU training (PyTorch DDP, e.g. via Kaggle / cloud GPU instances) used for wide
  hyperparameter and architecture sweeps of both networks ([#25](https://github.com/fortemate/dicechess-training/issues/25)).
- Profile and optimize the CPU-expand / GPU-evaluate batching pipeline and the training data
  loader.

**Model qualification** — train and qualify evaluation models on the manufactured dataset and
validate them in controlled arenas and on independent public play services.

Public artifacts land in this repository: the pipeline code, the sample and benchmark fixtures,
and open benchmark results. The production bots' tournament-tuned weights, opening books,
and configurations are outside its scope — the same open-core boundary described below.

## Workload shape

- **Tree expansion** is CPU-bound: headless JVM workers built on the open
  [dicechess-engine](https://github.com/fortemate/dicechess-engine) (Scala 3).
- **Leaf evaluation and exact rescoring** batch onto GPUs; **training** runs on GPUs
  (PyTorch, DDP for sweeps).
- **Serving** is CPU/edge: INT8 ONNX inside the engine on Cloudflare Workers, Cloud Run, and
  Raspberry Pi bots — the cluster manufactures artifacts; production never depends on it.

## Relationship to the Fortemate ecosystem

| Repository | Role |
| --- | --- |
| [dicechess-engine](https://github.com/fortemate/dicechess-engine) | Open engine: move generation and AI search (JVM / JS / Wasm) |
| [dicechess-play](https://github.com/fortemate/dicechess-play) / [dicechess-play-api](https://github.com/fortemate/dicechess-play-api) | Fortemate web application and API, published through [fortemate.com](https://fortemate.com) |
| **dicechess-training** (this repo) | Open training pipeline: label factory, training, distillation, evaluation |

Trained weights, opening books, and tournament-tuned bot configurations remain private,
following an open-core model: the framework is open; the competitive artifacts are not.

## License

[AGPL-3.0-only](LICENSE), consistent with the engine and the platform (SPDX `AGPL-3.0-only`; owner decision in fortemate/dicechess-engine#223).
