# dicechess-training

Open self-play data generation and model training pipeline for [Dice Chess](https://dicechess.com) — designed to scale from a single workstation GPU to an HPC cluster.

> **Status: scaffold under active extraction.** This repository is being ported from
> Fortemate's private, working training pipeline ahead of the
> [European AI Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00003yKxcX2AS/se000475)
> (EuroHPC / Open Hackathons, October 6–29, 2026). Porting roadmap:
>
> - [ ] Self-play worker and training-data schema — target mid-September 2026
> - [ ] Training loop, distillation, and evaluation harness — target end of September 2026
> - [ ] Runnable end-to-end single-node slice with a sample dataset — before the event

## Why

Dice Chess is a chess variant where three piece-type dice are rolled each turn, and only the
piece types shown may move. Fortemate builds the open-source engine and platform for it, plus
ML evaluation models (win-probability / expected value) that power the strongest bots on the
public [Glicko-2 rating ladder](https://github.com/fortemate/dicechess-play-api).

Today those models are trained from archived games and limited self-play generated on a
single consumer GPU. Self-play throughput is the bottleneck: stronger models need orders of
magnitude more high-quality positions than we can currently generate.

## Workload shape

- **Self-play generation** is CPU-bound tree search: headless JVM workers built on the open
  [dicechess-engine](https://github.com/fortemate/dicechess-engine) (Scala 3) play
  engine-vs-engine games in parallel and emit outcome-labeled training examples.
- **Training** runs on GPU: a PyTorch win-probability / EV network fit on self-play and
  archive data, with knowledge distillation to LightGBM and ONNX export for low-latency
  serving.
- **Evaluation** combines offline metrics (log-loss and calibration on games held out at the
  game level) with rated engine-vs-engine matches, and ultimately rated games against the
  public bot ladder.
- A hackathon direction we want mentor guidance on is **model-in-the-loop self-play**:
  batching concurrent position evaluations from many CPU search workers onto GPUs serving
  the ONNX network.

## Hackathon goals (October 2026)

**Primary goal** — parallelize self-play generation across cluster nodes, targeting ≥100×
the single-node baseline throughput (the baseline measurement lands here together with the
worker code).

**Stretch goals**

- Scale network training from one GPU to multi-GPU / multi-node (PyTorch DDP), sized for a
  larger teacher network and repeated retraining generations rather than a single fit.
- Profile GPU training with NVIDIA Nsight Systems and JVM self-play workers with
  async-profiler; eliminate the top bottlenecks identified.

**Post-event** — retrain the champion evaluation model on the enlarged dataset and validate
it with rated games on the public ladder.

Everything produced at the hackathon is public: the scaled pipeline code and the benchmark
and scaling results land in this repository.

## Relationship to the Fortemate ecosystem

| Repository | Role |
| --- | --- |
| [dicechess-engine](https://github.com/fortemate/dicechess-engine) | Open engine: move generation and AI search (JVM / JS / Wasm) |
| [dicechess-play](https://github.com/fortemate/dicechess-play) / [dicechess-play-api](https://github.com/fortemate/dicechess-play-api) | Public play platform and real-time server with an open Bot API |
| **dicechess-training** (this repo) | Open training pipeline: self-play, training, distillation, evaluation |

Trained weights, opening books, and tournament-tuned bot configurations remain private,
following an open-core model: the framework is open; the competitive artifacts are not.

## License

[AGPL-3.0](LICENSE), consistent with the engine and the platform.
