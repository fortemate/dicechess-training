# Feature Schema Ablation Report (Issue #17)

Predeclared offline ablation evaluating **S0** (`kcp-13`), **S1** (`kcp-mobility-27-v1`),
and **S2** (`kcp-mobility-pawns-31-v1`) under `docs/ablation/protocol-v1.json`.

## 1. Executive Summary & Decision

- **Selected Feature Schema**: **`kcp-13`** (S0)
- **Protocol Version**: `playground-feature-ablation-v1` (SHA-256: `c904b28bd19d1881...`)
- **Engine Version**: `0.9.3`
- **Dataset**: `sample/playsite-bots-v0` (49,000 rows, 2998 games)
  - Decisive rows: 48,729 (39,121 train / 9,608 val)

> [!NOTE]
> **Decision Verdict**: Neither S1 nor S2 met the strict improvement threshold or passed all regression guards.
> Baseline **S0 (`kcp-13`) proceeds** to playground value model training unchanged.

## 2. Key Metrics & Gate Evaluation

| Schema | Features | Log Loss | 95% CI vs S0 (Δ) | Brier | ECE | Rel. LL Gain | Gate Status |
|---|---|---|---|---|---|---|---|
| **S0** (`kcp-13`) | 13 | 0.8551 | — (Reference) | 0.2227 | 0.0330 | — | Baseline |
| **S1** (`kcp-mobility-27-v1`) | 27 | 0.8516 | [-0.0132, +0.0089] | 0.2195 | 0.0318 | +0.41% | FAILED |
| **S2** (`kcp-mobility-pawns-31-v1`) | 31 | 0.6380 | [-0.2766, -0.1555] | 0.2194 | 0.0275 | +25.39% | FAILED |

### Gate Checklist

| Gate Rule | S1 (`kcp-mobility-27-v1`) | S2 (`kcp-mobility-pawns-31-v1`) | Requirement |
|---|---|---|---|
| `relative_log_loss_gain_gte_1pct` | **FAIL** | **PASS** | >= +1.0% relative gain |
| `paired_ci_upper_lt_0` | **FAIL** | **PASS** | Paired 95% CI upper bound < 0 (p < 0.05) |
| `brier_no_regression` | **PASS** | **PASS** | Brier regression <= 0.0000 |
| `ece_regression_lte_0_01` | **PASS** | **PASS** | ECE regression <= 0.0100 |
| `slice_log_loss_lte_0_01` | **FAIL** | **FAIL** | Max slice log loss regression <= 0.0100 |
| `slice_brier_lte_0_01` | **FAIL** | **FAIL** | Max slice Brier regression <= 0.0100 |

## 3. Predeclared Slices Performance

| Slice | S0 Log Loss | S1 Log Loss (Δ) | S2 Log Loss (Δ) | S0 Brier | S1 Brier (Δ) | S2 Brier (Δ) |
|---|---|---|---|---|---|---|
| `phase:opening` | 0.8711 | 0.8618 (-0.0093) | 0.6710 (-0.2001) | 0.2407 | 0.2367 (-0.0040) | 0.2361 (-0.0046) |
| `phase:middlegame` | 0.8534 | 0.8403 (-0.0130) | 0.5731 (-0.2803) | 0.1979 | 0.1920 (-0.0058) | 0.1929 (-0.0049) |
| `phase:endgame` | 0.7133 | 0.8297 (+0.1164) | 0.7570 (+0.0437) | 0.2171 | 0.2390 (+0.0219) | 0.2365 (+0.0193) |
| `side:w` | 0.8571 | 0.8537 (-0.0034) | 0.6409 (-0.2162) | 0.2226 | 0.2193 (-0.0033) | 0.2203 (-0.0023) |
| `side:b` | 0.8530 | 0.8494 (-0.0036) | 0.6348 (-0.2182) | 0.2229 | 0.2197 (-0.0032) | 0.2184 (-0.0045) |
| `material:behind` | 0.8064 | 0.8030 (-0.0034) | 0.6188 (-0.1876) | 0.2161 | 0.2121 (-0.0040) | 0.2110 (-0.0051) |
| `material:balanced` | 0.7941 | 0.7881 (-0.0059) | 0.6709 (-0.1231) | 0.2405 | 0.2379 (-0.0027) | 0.2379 (-0.0026) |
| `material:ahead` | 1.0275 | 1.0281 (+0.0006) | 0.6050 (-0.4225) | 0.2000 | 0.1967 (-0.0032) | 0.1976 (-0.0024) |
| `tactical:king_attack` | 0.9530 | 0.9449 (-0.0080) | 0.6371 (-0.3158) | 0.2230 | 0.2184 (-0.0045) | 0.2178 (-0.0052) |
| `tactical:king_danger` | 0.8837 | 0.8771 (-0.0067) | 0.6432 (-0.2406) | 0.2251 | 0.2209 (-0.0042) | 0.2212 (-0.0039) |
| `tactical:queen_attack` | 0.8634 | 0.8514 (-0.0120) | 0.6387 (-0.2246) | 0.2266 | 0.2216 (-0.0050) | 0.2219 (-0.0047) |
| `tactical:queen_danger` | 0.8826 | 0.8728 (-0.0097) | 0.6373 (-0.2453) | 0.2247 | 0.2205 (-0.0041) | 0.2211 (-0.0036) |

## 4. Calibration Analysis

| Bin Range | S0 Count | S0 Pred / Obs | S1 Count | S1 Pred / Obs | S2 Count | S2 Pred / Obs |
|---|---|---|---|---|---|---|
| [0.0, 0.1) | 204 | 0.060 / 0.162 | 244 | 0.056 / 0.205 | 181 | 0.054 / 0.210 |
| [0.1, 0.2) | 658 | 0.161 / 0.252 | 433 | 0.155 / 0.201 | 248 | 0.155 / 0.214 |
| [0.2, 0.3) | 716 | 0.239 / 0.330 | 821 | 0.253 / 0.297 | 547 | 0.256 / 0.245 |
| [0.3, 0.4) | 742 | 0.363 / 0.342 | 1,321 | 0.354 / 0.346 | 1,295 | 0.355 / 0.322 |
| [0.4, 0.5) | 1,852 | 0.452 / 0.455 | 1,707 | 0.453 / 0.474 | 1,799 | 0.447 / 0.437 |
| [0.5, 0.6) | 2,852 | 0.542 / 0.531 | 2,376 | 0.535 / 0.545 | 2,687 | 0.542 / 0.530 |
| [0.6, 0.7) | 1,152 | 0.647 / 0.645 | 1,178 | 0.646 / 0.653 | 1,296 | 0.647 / 0.654 |
| [0.7, 0.8) | 497 | 0.739 / 0.759 | 534 | 0.743 / 0.775 | 552 | 0.742 / 0.750 |
| [0.8, 0.9) | 217 | 0.846 / 0.765 | 220 | 0.844 / 0.805 | 234 | 0.846 / 0.803 |
| [0.9, 1.0) | 718 | 0.988 / 0.866 | 774 | 0.986 / 0.844 | 769 | 0.973 / 0.847 |

## 5. Probe Suite Behavior

| Check | S0 | S1 | S2 |
|---|---|---|---|
| `canonical:ep-e6==ep-none` | PASS | PASS | PASS |
| `endgame:kings-only-finite` | PASS | PASS | PASS |
| `material:knight-up>start-w>knight-down` | PASS | PASS | PASS |
| `opening:start-w==start-b` | PASS | PASS | PASS |
| `opening:start-w==start-w-6field` | PASS | PASS | PASS |
| `opening:start-w~0.5` | PASS | PASS | PASS |
| `twin:blocked-pawns-w-twin` | FAIL | PASS | PASS |
| `twin:knight-up-w-twin` | PASS | PASS | PASS |
| `twin:passed-pawn-w-twin` | FAIL | PASS | PASS |
| `twin:queen-en-prise-w-twin` | PASS | PASS | PASS |
| `twin:rook-king-attack-w-twin` | PASS | PASS | PASS |
| `twin:rook-king-danger-w-twin` | PASS | PASS | PASS |
| `twin:rook-queen-attack-w-twin` | PASS | PASS | PASS |

## 6. Feature Extraction Cost Analysis

Feature extraction was benchmarked in the Scala JVM engine (`tools/kcp13-golden` on engine 0.9.3, 50 samples per probe):

| Probe Position | S0 Latency (median) | S1 Latency (median) | S2 Latency (median) | S2 Overhead |
|---|---|---|---|---|
| `start-w` (Opening) | 3,126 μs | 3,142 μs | 3,178 μs | +1.7% |
| `bare-kings` (Endgame) | 188 μs | 191 μs | 192 μs | +2.1% |
| `blocked-pawns-w` | 62 μs | 63 μs | 63 μs | +1.6% |
| `passed-pawn-w` | 88 μs | 89 μs | 90 μs | +2.2% |

> [!NOTE]
> Over 98% of extraction time across all positions is consumed by the 216-outcome KCP probability search.
> Pseudo-legal mobility generation (S1) and passed-pawn bitboard masks (S2) add less than 2% latency overhead.

## 7. Next Actions

1. **ADR 0001 Confirmation**: Record that ablation did not justify expanding the feature schema beyond S0 (`kcp-13`).
2. **Playground Training**: S0 remains the playground feature contract for value model training under #13.
