# dicechess-training

Open training pipeline for [Dice Chess](https://dicechess.com) — a GPU **label factory** that manufactures deep exact-search labels, and two small networks trained on them, designed to scale from a single workstation to an HPC cluster.

> **Status: scaffold under active extraction.** This repository is being built ahead of the
> [European AI Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00003yKxcX2AS/se000475)
> (EuroHPC / Open Hackathons, October 6–29, 2026). September roadmap, prepared entirely
> without cluster access:
>
> - [ ] Star2 chance-node pruning (depth 3) + transposition table in the open engine, gated
>       by a depth-3 vs depth-2 arena experiment — mid-September 2026
> - [ ] First PyTorch → ONNX training stack, validated end-to-end at toy scale — mid-September 2026
> - [ ] Engine hooks: ONNX evaluator at chance nodes and a learned pre-ranker slot — end of September 2026
> - [ ] CPU label-factory v0, data schema, and a seed dataset — before the event

## Why

Dice Chess is a chess variant where three piece-type dice are rolled each turn, and only the
piece types shown may move. Every turn therefore passes through a chance node with up to 216
outcomes (56 distinct multisets), and that explosion shapes everything we have measured while
building the engine, the platform, and the bots:

- **Search beats features.** Richer evaluation features gained +20 percentage points at
  1-ply search and nothing at 2-ply. Playing strength lives in depth.
- **Depth is capped at 2 plies** by the chance-node explosion. Star1/Star2 pruning is
  designed but not yet implemented in the engine.
- **More data of the same quality no longer helps:** ≤1 pp per doubling of supervised
  training data from archived games. Better labels are needed, not more labels.
- **Our strongest bot is not the ML one.** A hand-crafted evaluation with an exact 216-roll
  rescoring phase beats our best model-based bot 66.4% head-to-head — and under real time
  budgets it only manages to exactly rescore 1–2 candidate moves. A learned move pre-ranker
  already measured **+4.8 pp** from improving which candidates get rescored.

The conclusion drives this repository: use HPC to manufacture **deep exact labels** that our
own hardware can never compute, and compress them into **small networks** that run at the
edge.

## The plan: a label factory and two nets

**Label factory** (the cluster-bound workload): CPU workers expand Star2-pruned depth-3
search trees over positions sampled from ~2M archived games and self-play; GPUs evaluate
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

## Hackathon goals (October 2026)

**Primary** — run the label factory at cluster scale (target: 100M+ depth-3-labeled
positions) and train both networks on it, with labeling throughput and scaling measured and
profiled with the mentors.

**Stretch**

- Multi-GPU / multi-node training (PyTorch DDP) used for wide hyperparameter and
  architecture sweeps of both nets.
- Profile and optimize the CPU-expand / GPU-evaluate batching pipeline and the training data
  loader (NVIDIA Nsight Systems on the GPU side, async-profiler on the JVM side).

**Post-event** — retrain the production evaluation model on the manufactured dataset and
validate with rated games on the public ladder.

Everything produced at the hackathon is public: the pipeline code, the labeled dataset, and
the benchmark and scaling results land in this repository.

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
| [dicechess-play](https://github.com/fortemate/dicechess-play) / [dicechess-play-api](https://github.com/fortemate/dicechess-play-api) | Public play platform and real-time server with an open Bot API |
| **dicechess-training** (this repo) | Open training pipeline: label factory, training, distillation, evaluation |

Trained weights, opening books, and tournament-tuned bot configurations remain private,
following an open-core model: the framework is open; the competitive artifacts are not.

## License

[AGPL-3.0](LICENSE), consistent with the engine and the platform.
