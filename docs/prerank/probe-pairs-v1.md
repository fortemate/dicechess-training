# The probe-pair gate

Carried from `dicechess-ev#1`. On authored pairs that differ only in whether the mover's queen is
left en prise, does the ranker put the blunder below its safe twin?

**It answers a different question than the one it was written to ask, and the answer matters
more.** The schema a pre-ranker can afford has no feature for a piece being attacked, so no
`rich-9` model can pass this gate by seeing safety. Both the learned ranker and the model in
production score 5 of 8 — and fail the same three.

| ordering                                          |      gate | fails                         |
| ------------------------------------------------- | --------: | ----------------------------- |
| `material_diff` — the engine's shipped pre-ranker | **0 / 8** | every pair                    |
| `oracle-3` — what `dexus-atlas-1` serves          |     5 / 8 | the three rook-attacker pairs |
| the learned listwise pre-ranker                   |     5 / 8 | the same three                |

## Why material scores zero

Not because it ranks badly. Because `material_diff` is **identical in both halves of every pair** —
6.0 against 6.0, 8.0 against 8.0. A queen that is attacked and a queen that is safe have the same
material, so the ordering the engine ships is _exactly indifferent_ between a blunder and the move
that avoids it. A tie is counted as a failure here, because the seam keeps the top `k` and an
ordering with no preference has expressed none.

That is the complaint this whole line of work started from, demonstrated rather than asserted: the
material pre-ranker cannot see safety at all.

## Why both models score 5 of 8, and fail the same three

`kcp-13`'s last four columns are the capture probabilities. **`rich-9` is exactly `kcp-13` without
them.** So within every certified pair the nine columns the model sees are identical except one:

| pair                  | attacker | `mobility_diff` blunder → safe | mobility favours |
| --------------------- | -------- | -----------------------------: | ---------------- |
| pawn-d4, pawn-d4-twin | pawn     |                        22 → 26 | the safe twin    |
| pawn-e4               | pawn     |                        21 → 25 | the safe twin    |
| knight-d3             | knight   |                        17 → 21 | the safe twin    |
| bishop-d4             | bishop   |                        14 → 16 | the safe twin    |
| rook-d4, rook-d4-twin | rook     |                        14 → 11 | **the blunder**  |
| rook-e5               | rook     |                        12 → 10 | **the blunder**  |

A model's answer on a pair is therefore entirely determined by how it responds to that one number
with the other eight held fixed. Mobility is a _correlate_ of safety here — a piece on an active
square is both more mobile and more exposed — and it points the right way five times and the wrong
way three.

So the 5 of 8 is not the model half-learning safety. It is mobility being a proxy that happens to
be right five times in eight, and the two independently-trained models agreeing because they are
reading the same proxy. **A rich-9 artifact cannot pass this gate for the right reason.**

## What the pairs are, and why they are certified rather than asserted

Each pair is two positions with identical material and an identical placement of the mover's own
pieces. Only the opponent's attacker moves: in one it attacks the queen, in the other it does not.

Three of the first six pairs written by hand were wrong — an attacker that did not attack, a
"safe" square equally reachable in a three-die turn, and one where the attacking rook also bore on
the mover's king and so moved `king_safety_diff` by 2000. So each pair carries a second feature
vector under `kcp-13`, and is admitted only when `queen_capture_danger` is at least 0.2 higher in
the blunder. The observed margins are 0.35 to 0.42 against residuals of 0.00 to 0.08 — a "safe"
twin is rarely at exactly zero, because an opponent can often manoeuvre into range within a turn.

`certify()` runs before any model is scored, so a fixture that stops being a fixture fails loudly
instead of passing everything.

## Running it

```bash
uv run python -m dicechess_training.prerank probe-pairs <model.onnx>
```

Exit code 2 when any pair fails, so it can be used as a gate. It prints the failing pairs together
with what `rich-9` could see of them, because a bare pass rate from this fixture is misleading
without that.

## What this means for #9

The DoD asks that the ranker place the blunder below its safe twin. **On `rich-9` that cannot be
satisfied for the right reason**, and this is a schema finding rather than a training one:

- the learned ranker is a clear improvement on what the engine ships — 0 of 8 to 5 of 8 — and
  identical to what production already serves;
- the remaining three are not a training failure to fix with more data or more epochs. They are
  the schema's blind spot, and more of the same corpus will not move them;
- the columns that would fix it are the four `kcp-13` drops, and they were dropped for cost: the
  engine's own documentation says `kcp-13` "is not viable at this seam at all" at the width the
  pass runs.

So the decision this hands back is a real one: accept a pre-ranker that reads a proxy for safety,
or find a safety feature cheap enough for a pass that sees every legal turn. Neither is settled
here, and nothing in `results-v1.md` should be read as having settled it.
