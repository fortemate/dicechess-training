# Playground Feature Schema Ablation Protocol

Status: **FROZEN**. Committed prior to generating results in Issue #17.

## Context

Parent epic: [Issue #12](https://github.com/fortemate/dicechess-training/issues/12).
Decision record: [ADR 0001](../decisions/0001-playground-train-serve-contract.md).
Benchmark specification: [Benchmark v1](../benchmark/README.md).

The evaluation service serves `standard-kcp` over the `kcp-13` feature schema. Owner research indicates that position properties such as expected wasted rolls, pawn blockage, tempo as independent own/opponent mobility, and passed pawns are critical for one-ply position evaluation. The engine implements these features as versioned extractors (`kcp-mobility-27-v1` and `kcp-mobility-pawns-31-v1`) in `com.fortemate:dicechess-engine_3:0.9.3` (engine Issue #215).

This protocol defines the predeclared offline ablation to select the feature schema for the first real playground model.

## Candidate Schemas

| Schema | Identifier | Columns | Description |
| --- | --- | ---: | --- |
| **S0** | `kcp-13` | 13 | Served baseline: 7 material, `mobility_diff`, `king_safety_diff`, 4 capture probabilities |
| **S1** | `kcp-mobility-27-v1` | 27 | S0 + pseudo-legal move counts per moving piece type for both sides (`own_moves_p..k`, `opp_moves_p..k`, 12 cols) + piece diversity indices (`own_pdi`, `opp_pdi`, 2 cols) |
| **S2** | `kcp-mobility-pawns-31-v1` | 31 | S1 + passed-pawn counts (`own_passed_pawns`, `opp_passed_pawns`, 2 cols) + max passed-pawn ranks (`own_passed_max_rank`, `opp_passed_max_rank`, 2 cols) |

## Model Family & Training Hyperparameters

- **Architecture**: PyTorch MLP with identical capacity across all schemas:
  - Input layer: dimension matching schema width ($N \in \{13, 27, 31\}$)
  - Hidden layers: 2 dense layers of 64 units each with ReLU activations
  - Output layer: 1 linear unit with Sigmoid activation ($\hat{y} \in [0, 1]$)
- **Loss function**: Binary cross-entropy (log-loss)
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.999$), learning rate $\eta = 0.001$, batch size 256
- **Epochs**: 5
- **Seeds**: Exactly 5 predeclared seeds: `[11, 23, 47, 89, 131]`

## Dataset & Split Policy

- **Corpus**: `sample/playsite-bots-v0` (49,000 positions across 2,999 games).
- **Target**: Decisive game outcome from the side-to-move perspective (win = 1.0, loss = 0.0). Draws (0.5) are counted and excluded from binary scores.
- **Game-level split**: `int(sha256('playground-v1:' + game_id), 16) % 10000`:
  - Train: value $< 8000$ (80% of games)
  - Validation: value $\ge 8000$ (20% of games)
- No position leakage across splits.

## Evaluation & Metrics

- **Primary metric**: Log-loss $-\frac{1}{N}\sum [y \ln(p) + (1-y) \ln(1-p)]$, clipped at $10^{-15}$.
- **Calibration metrics**: Brier score $\frac{1}{N}\sum (p - y)^2$ and Expected Calibration Error (ECE) across 10 equal-width bins.
- **Uncertainty**: 95% bootstrap confidence intervals computed by resampling whole games/groups (1,000 resamples, seed 13). Paired deltas against S0 computed on identical resample draws.
- **Slices**:
  - Phase: opening ($\text{ply} \le 10$), endgame ($\text{total\_material} \le 20$), middlegame (otherwise).
  - Side to move: White (`w`), Black (`b`).
  - Material balance: behind ($\text{material\_diff} < -1$), balanced ($-1 \le \text{material\_diff} \le 1$), ahead ($\text{material\_diff} > 1$).
  - Tactical flags: positions with active king or queen capture threats.
- **Golden probe suite**: Invariant checks against engine-computed vectors (`tests/fixtures/`):
  - Initial position symmetry (`start-w` vs `start-b`)
  - En passant and clock canonicalization
  - Monotonic material order
  - Blocked pawn chain (`blocked-pawns-w`)
  - Passed pawn advancement (`passed-pawn-w`)
  - Mover-canonical twin equality across all probes

## Decision Gate (frozen in #13)

A wider candidate (S1 or S2) is selected over S0 iff:
1. Relative log-loss improvement on validation: $\frac{\text{LL}_{S0} - \text{LL}_S}{\text{LL}_{S0}} \ge 1.0\%$.
2. Paired 95% bootstrap CI upper bound of log-loss delta: $\Delta \text{LL}_{97.5} < 0$.
3. Brier score does not regress: $\text{Brier}_S - \text{Brier}_{S0} \le 0.0$.
4. ECE does not regress beyond tolerance: $\text{ECE}_S - \text{ECE}_{S0} \le 0.01$.
5. Critical slices do not regress beyond point-estimate tolerance: $\Delta \text{LL}_{\text{slice}} \le 0.01$, $\Delta \text{Brier}_{\text{slice}} \le 0.01$.
6. Serving extraction cost measured on JVM is bounded (latency overhead <= 10% relative to S0 across benchmark probes).

If S1 or S2 clears the gate, the best-performing qualifying schema is selected and a `dicechess-evaluation` issue is opened.
If neither clears the gate, S0 is retained as the playground model schema.
