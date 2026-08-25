# Sample dataset

`playsite-bots-v0/` — **49,000 positions from 2,999 games**, schema v0 Parquet shards (~1.1 MB).
Committed so that `mise run demo` trains on real Dice Chess positions out of the box.

## Provenance

Every game here was played **bot against bot on Fortemate's own platform** — both sides are
Fortemate house bots (engine experiments, ladder anchors, and evaluation-model bots), and the
games were recorded by our own server. No human games are included, and no third-party data:
the corpus is entirely ours to publish.

Selection: games from the platform's own game source with a recorded outcome and both players
of type `bot`, sampled deterministically (by hash of the game id) rather than by recency, so
the sample is not skewed toward whichever bots happened to play last. Positions follow the
same turn-level guards the analytics export uses — no-op self-loops and abandoned partial
turns dropped, legal passes and terminal king-captures kept.

The full archive (millions of games, including human play) stays private; this sample is a
deliberate, self-contained slice for the open pipeline.

## What one row means

| Column    | Meaning                                                                      |
| --------- | ---------------------------------------------------------------------------- |
| `game_id` | Game identifier — the unit of the train/holdout split                        |
| `ply`     | Turn number within the game (1-based, as recorded)                           |
| `fen`     | Position before the turn: placement, active color, castling, en passant      |
| `dice`    | The three piece-type dice for this turn, upper-case, order-insensitive       |
| `side`    | `w` or `b` — who is to move                                                  |
| `result`  | Game outcome **from the mover's perspective**: 1.0 win, 0.5 draw, 0.0 loss   |

Class balance: 25,149 wins / 23,580 losses / 271 draws — draws are genuinely rare in Dice
Chess, which is why calibration matters more than accuracy.

## What a good result on it looks like

`mise run demo` trains the value net on 80% of the games and reports holdout metrics against
a **no-information baseline** — the log-loss of always predicting the base rate (0.693 for a
balanced set). The shipped config (32 hidden units, 2 epochs) lands around **0.646 vs a 0.693
baseline**: a real but modest edge, which is what a raw board plus dice can buy with no search
at all.

The full-size 774→256→256→1 net does **worse** here — it memorizes 40k positions within two
epochs and comes out above the baseline, systematically overconfident (predicting 0.96 where
the true rate is 0.80). That is not a bug to fix in the architecture; it is the program's
premise in miniature: at this data scale **the labels bind, not the model**. Deeper, exact
labels are what the hackathon's label factory manufactures.

## Regenerating or extending

`dicechess_training.ingest.convert_export_file` converts a `dicechess-analytics` training
export (CSV or CSV.gz) into these shards, translating the export's White-POV `1/0/-1` result
encoding into the mover-perspective 0.0/0.5/1.0 of schema v0 and normalizing dice case.

```python
from dicechess_training import ingest

ingest.convert_export_file("training_data.csv.gz", "sample/playsite-bots-v0")
```

Bigger datasets belong in `data/` (git-ignored), not here.
