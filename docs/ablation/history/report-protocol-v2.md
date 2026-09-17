# Feature Schema Ablation Report (Issue #17)

> [!WARNING]
> **Superseded, retained for provenance.** This run used protocol v2, whose recipe trained on raw
> feature columns with `BCELoss` on a saturating sigmoid. Under the no-information floor added in
> protocol v4 (Issue #27) none of its three arms is admissible: every schema scored a worse
> validation log loss than the constant train-base-rate predictor (0.6926), and the baseline's
> seed spread (±0.4588) was far larger than the +1 % relative threshold the gate had to resolve.
> Its selection of S0 is not evidence. The current report is [../report.md](../report.md).

Predeclared offline ablation evaluating **S0** (`kcp-13`), **S1** (`kcp-mobility-27-v1`), and **S2** (`kcp-mobility-pawns-31-v1`) under `docs/ablation/protocol-v2.json` (amends `protocol-v1.json`).

> [!NOTE]
> **Provisional Development Report**: Evaluated on public sample `sample/playsite-bots-v0` (development-only; ineligible for benchmark qualification without separately reviewed data-use evidence).
> Final qualification requires owner-run evaluation of the frozen private corpus under the same protocol.
> Reference: **Private Decision: Playground Feature Schema Qualification (Issue #17)**.

## 1. Executive Summary & Decision

- **Selected Feature Schema**: **`kcp-13`** (S0)
- **Protocol Version**: `playground-feature-ablation-v2` (SHA-256: `0cb0f0b3a64c117a...`)
- **Engine Version**: `0.9.3`
- **Dataset**: `sample/playsite-bots-v0` (49,000 rows, 2998 games)
  - Decisive rows: 48,729 (39,121 train / 4,628 val / 4,980 test holdout)
  - Canonical position leakage: train:val = 238, train:test = 253, val:test = 117 (unseen val positions: 3,805)

> [!NOTE]
> **Development Verdict**: Neither S1 nor S2 met the strict improvement threshold or passed all regression guards.
> Baseline **S0 (`kcp-13`) proceeds** to playground value model training unchanged.

### Input Shard Provenance

| Schema | Shard File | SHA-256 Digest |
|---|---|---|
| **S0** | `playsite-bots-sample-000.parquet` | `f23b58e6b82c3438...` |
| **S0** | `playsite-bots-sample-001.parquet` | `32b83aaec8406bc8...` |
| **S1** | `playsite-bots-sample-000.parquet` | `8ee5131f62aadbbb...` |
| **S1** | `playsite-bots-sample-001.parquet` | `19fdebaac8987307...` |
| **S2** | `playsite-bots-sample-000.parquet` | `1a6b8de7e8f0f15f...` |
| **S2** | `playsite-bots-sample-001.parquet` | `9e5267262b95c417...` |

## 2. Primary Estimand: Single-Model Replication

Evaluation of single-model performance across independent random seeds (mean ± std). Paired 95% confidence intervals are computed via whole-game bootstrap resampling on the mean seed loss delta of the primary estimand:

| Schema | Features | Log Loss | 95% CI vs S0 (Δ) | Brier | ECE | Rel. LL Gain | Gate Status |
|---|---|---|---|---|---|---|---|
| **S0** (`kcp-13`) | 13 | 1.1366 ± 0.4588 | — (Reference) | 0.2388 ± 0.0044 | 0.0653 ± 0.0107 | — | Baseline |
| **S1** (`kcp-mobility-27-v1`) | 27 | 0.8921 ± 0.0108 | [-0.2931, -0.1914] | 0.2314 ± 0.0025 | 0.0500 ± 0.0099 | +21.51% | FAILED |
| **S2** (`kcp-mobility-pawns-31-v1`) | 31 | 0.7991 ± 0.1042 | [-0.4190, -0.2612] | 0.2308 ± 0.0020 | 0.0568 ± 0.0066 | +29.69% | FAILED |

## 3. Unseen Validation Positions Performance

Evaluation on validation positions with no FEN/side overlap in the training partition:

| Schema | Unseen Positions | Log Loss | 95% CI vs S0 (Δ) | Brier | ECE | Unseen Gate |
|---|---|---|---|---|---|---|
| **S0** (`kcp-13`) | 3,805 | 1.2192 ± 0.5447 | — (Reference) | 0.2372 ± 0.0050 | 0.0763 ± 0.0097 | Baseline |
| **S1** (`kcp-mobility-27-v1`) | 3,805 | 0.9297 ± 0.0132 | [-0.3503, -0.2265] | 0.2288 ± 0.0031 | 0.0566 ± 0.0140 | **PASS** |
| **S2** (`kcp-mobility-pawns-31-v1`) | 3,805 | 0.8196 ± 0.1230 | [-0.4894, -0.3159] | 0.2280 ± 0.0022 | 0.0635 ± 0.0057 | **PASS** |

### 2b. Secondary Diagnostic: 5-Model Ensemble

> [!NOTE]
> Ensemble predictions (average probability across 5 seeds). Reported for variance-reduction diagnostics; not the single-model deployable contract.

| Schema | Features | Ensemble Log Loss | Ensemble Brier | Ensemble ECE |
|---|---|---|---|---|
| **S0** (`kcp-13`) | 13 | 0.8791 | 0.2302 | 0.0404 |
| **S1** (`kcp-mobility-27-v1`) | 27 | 0.8823 | 0.2288 | 0.0460 |
| **S2** (`kcp-mobility-pawns-31-v1`) | 31 | 0.6591 | 0.2272 | 0.0326 |

### Gate Checklist

| Gate Rule | S1 (`kcp-mobility-27-v1`) | S2 (`kcp-mobility-pawns-31-v1`) | Requirement |
|---|---|---|---|
| `relative_log_loss_gain_gte_1pct` | **PASS** | **PASS** | >= +1.0% relative gain on full validation |
| `paired_ci_upper_lt_0` | **PASS** | **PASS** | Paired 95% group-bootstrap CI upper bound < 0 on mean seed delta |
| `brier_no_regression` | **PASS** | **PASS** | Brier regression <= 0.0000 |
| `ece_regression_lte_0_01` | **PASS** | **PASS** | ECE regression <= 0.0100 |
| `slice_log_loss_lte_0_01` | **FAIL** | **FAIL** | Max slice log loss regression <= 0.0100 |
| `slice_brier_lte_0_01` | **FAIL** | **FAIL** | Max slice Brier regression <= 0.0100 |
| `unseen_positions_exist` | **PASS** | **PASS** | Non-empty unseen validation partition (no-position-leakage) |
| `unseen_log_loss_lte_0_01` | **PASS** | **PASS** | Unseen positions log loss regression <= 0.0100 |
| `unseen_brier_no_regression` | **PASS** | **PASS** | Unseen positions Brier regression <= 0.0000 |
| `extraction_cost_evidence_valid` | **PASS** | **PASS** | Verified JVM extraction benchmark artifact present |
| `extraction_cost_lte_tolerance` | **PASS** | **PASS** | Mean JVM extraction latency overhead <= 10.0% relative to S0 |

## 4. Predeclared Slices Performance (Single-Model Means)

| Slice | S0 Log Loss | S1 Log Loss (Δ) | S2 Log Loss (Δ) | S0 Brier | S1 Brier (Δ) | S2 Brier (Δ) |
|---|---|---|---|---|---|---|
| `phase:opening` | 1.1546 | 0.9113 (-0.2433) | 0.8189 (-0.3356) | 0.2502 | 0.2422 (-0.0081) | 0.2415 (-0.0087) |
| `phase:middlegame` | 1.1557 | 0.8449 (-0.3108) | 0.7516 (-0.4041) | 0.2206 | 0.2081 (-0.0125) | 0.2092 (-0.0115) |
| `phase:endgame` | 0.8799 | 1.0186 (+0.1387) | 0.9213 (+0.0414) | 0.2543 | 0.2823 (+0.0280) | 0.2729 (+0.0186) |
| `side:w` | 1.0950 | 0.8732 (-0.2218) | 0.7878 (-0.3071) | 0.2371 | 0.2302 (-0.0069) | 0.2297 (-0.0073) |
| `side:b` | 1.1811 | 0.9124 (-0.2687) | 0.8112 (-0.3699) | 0.2406 | 0.2326 (-0.0080) | 0.2319 (-0.0087) |
| `material:behind` | 1.1730 | 0.7951 (-0.3779) | 0.7327 (-0.4402) | 0.2397 | 0.2281 (-0.0116) | 0.2270 (-0.0126) |
| `material:balanced` | 0.9932 | 0.8240 (-0.1692) | 0.7647 (-0.2286) | 0.2473 | 0.2418 (-0.0055) | 0.2411 (-0.0063) |
| `material:ahead` | 1.3289 | 1.1338 (-0.1951) | 0.9441 (-0.3848) | 0.2232 | 0.2182 (-0.0050) | 0.2185 (-0.0047) |
| `tactical:king_attack` | 1.2807 | 0.9873 (-0.2934) | 0.8564 (-0.4243) | 0.2396 | 0.2301 (-0.0094) | 0.2293 (-0.0103) |
| `tactical:king_danger` | 1.2088 | 0.8869 (-0.3219) | 0.7974 (-0.4114) | 0.2423 | 0.2319 (-0.0104) | 0.2322 (-0.0101) |
| `tactical:queen_attack` | 1.2218 | 0.9038 (-0.3180) | 0.8078 (-0.4140) | 0.2429 | 0.2321 (-0.0109) | 0.2327 (-0.0102) |
| `tactical:queen_danger` | 1.2062 | 0.9289 (-0.2772) | 0.8232 (-0.3830) | 0.2412 | 0.2320 (-0.0092) | 0.2324 (-0.0089) |

## 5. Calibration Analysis (Ensemble Diagnostic)

| Bin Range | S0 Count | S0 Pred / Obs | S1 Count | S1 Pred / Obs | S2 Count | S2 Pred / Obs |
|---|---|---|---|---|---|---|
| [0.0, 0.1) | 99 | 0.055 / 0.192 | 120 | 0.053 / 0.283 | 85 | 0.053 / 0.282 |
| [0.1, 0.2) | 317 | 0.161 / 0.300 | 203 | 0.157 / 0.256 | 119 | 0.158 / 0.303 |
| [0.2, 0.3) | 355 | 0.239 / 0.355 | 412 | 0.252 / 0.313 | 266 | 0.256 / 0.278 |
| [0.3, 0.4) | 349 | 0.363 / 0.367 | 622 | 0.355 / 0.338 | 629 | 0.356 / 0.313 |
| [0.4, 0.5) | 903 | 0.452 / 0.445 | 820 | 0.453 / 0.491 | 868 | 0.447 / 0.444 |
| [0.5, 0.6) | 1,360 | 0.542 / 0.532 | 1,155 | 0.536 / 0.548 | 1,279 | 0.543 / 0.539 |
| [0.6, 0.7) | 576 | 0.646 / 0.648 | 577 | 0.646 / 0.638 | 646 | 0.646 / 0.644 |
| [0.7, 0.8) | 237 | 0.737 / 0.734 | 255 | 0.742 / 0.757 | 270 | 0.741 / 0.726 |
| [0.8, 0.9) | 106 | 0.849 / 0.689 | 109 | 0.844 / 0.725 | 117 | 0.846 / 0.726 |
| [0.9, 1.0) | 326 | 0.989 / 0.840 | 355 | 0.987 / 0.808 | 349 | 0.974 / 0.817 |

## 6. Probe Suite Behavior

| Check | S0 | S1 | S2 |
|---|---|---|---|
| `canonical:ep-e6==ep-none` | PASS | PASS | PASS |
| `endgame:kings-only-finite` | PASS | PASS | PASS |
| `material:knight-up>start-w>knight-down` | PASS | PASS | PASS |
| `opening:start-w==start-b` | PASS | PASS | PASS |
| `opening:start-w==start-w-6field` | PASS | PASS | PASS |
| `opening:start-w~0.5` | PASS | PASS | PASS |
| `twin:blocked-pawns-w-twin` | PASS | PASS | PASS |
| `twin:knight-up-w-twin` | PASS | PASS | PASS |
| `twin:passed-pawn-w-twin` | PASS | PASS | PASS |
| `twin:queen-en-prise-w-twin` | PASS | PASS | PASS |
| `twin:rook-king-attack-w-twin` | PASS | PASS | PASS |
| `twin:rook-king-danger-w-twin` | PASS | PASS | PASS |
| `twin:rook-queen-attack-w-twin` | PASS | PASS | PASS |

## 7. Feature Extraction Cost Analysis

Feature extraction measured with JVM engine `com.fortemate:dicechess-engine_3:0.9.3` (50 samples per probe after 20 warmups):
- Runtime: Java `26.0.2.1` (Homebrew) on `Mac OS X` (aarch64)
- Timestamp: `2026-09-14T11:01:31.708740Z`

| Probe Position | S0 Latency (median) | S1 Latency (median) | S2 Latency (median) | S2 Overhead |
|---|---|---|---|---|
| `start-w` | 3,184.5 μs | 3,056.8 μs | 2,870.0 μs | -9.9% |
| `start-b` | 2,836.3 μs | 2,864.0 μs | 2,852.7 μs | +0.6% |
| `blocked-pawns-w` | 54.1 μs | 54.6 μs | 55.0 μs | +1.7% |
| `passed-pawn-w` | 89.5 μs | 89.6 μs | 91.4 μs | +2.1% |

> [!NOTE]
> Over 98% of extraction time across all positions is consumed by the 216-outcome KCP probability search.
> Pseudo-legal mobility generation (S1) and passed-pawn bitboard masks (S2) add negligible latency overhead (<= 10% bound satisfied).

## 8. Next Actions

1. **Private Qualification**: Issue #17 remains open pending owner execution of the frozen protocol on the private corpus.
2. **ADR 0001 Maintenance**: Retain provisional development findings in ADR 0001; final amendment occurs after owner qualification.
3. **Value Model Training**: S0 remains the current baseline contract for #13.
