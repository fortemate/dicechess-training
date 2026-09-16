# Feature Schema Ablation Report (Issue #17)

Offline ablation evaluating **S0** (`kcp-13`), **S1** (`kcp-mobility-27-v1`), and **S2** (`kcp-mobility-pawns-31-v1`) under the protocol identified below.

> [!NOTE]
> **Provisional Development Report**: Input shard digests below identify the evaluated data. Development gate results do not establish final qualification.
> Final qualification requires owner-run evaluation of the frozen private corpus under the same protocol.
> Reference: **Private Decision: Playground Feature Schema Qualification (Issue #17)**.

## 1. Executive Summary & Decision

- **Selected Feature Schema**: **`kcp-13`** (S0)
- **Protocol Version**: `playground-feature-ablation-v4` (SHA-256: `96decbc3e6d49fd4...`)
- **Engine Version**: `0.9.3`
- **Dataset**: input enriched shards (49,000 rows, 2998 games)
  - Decisive rows: 48,729 (39,121 train / 4,628 val / 4,980 test holdout)
  - Canonical position leakage: train:val = 238, train:test = 253, val:test = 117 (unseen val positions: 3,805)

> [!NOTE]
> **Development Verdict**: Neither wider candidate cleared all development gates.
> Baseline **S0 (`kcp-13`) remains the development reference**; final owner qualification remains pending.

## 1b. Admissibility Against the No-Information Reference

The benchmark's default reference is a constant predictor. A schema whose mean single model scores no better than it has measured nothing and cannot be selected, whatever its relative comparison with S0 shows. Reference: `constant-train-base-rate` = 0.5162.

| Predictor | Log Loss | Brier | ECE | Verdict |
|---|---|---|---|---|
| No-information reference | 0.6926 | 0.2497 | 0.0002 | Reference |
| S0 (`kcp-13`) | 0.6385 | 0.2221 | 0.0350 | **admissible** (+0.0541) |
| S1 (`kcp-mobility-27-v1`) | 0.6497 | 0.2239 | 0.0408 | **admissible** (+0.0429) |
| S2 (`kcp-mobility-pawns-31-v1`) | 0.6443 | 0.2233 | 0.0391 | **admissible** (+0.0483) |

Recipe: feature standardisation `train-statistics`, epoch budget selected on the inner tuning split (seed 11: 5, seed 23: 5, seed 47: 5, seed 89: 10, seed 131: 5).

### Input Shard Provenance

The file digest identifies the artifact this run read. The content digest covers the semantic metadata and the column values in a canonical order, and is what an independent re-run compares: the Parquet writer orders its footer encoding sets differently between producer sessions, so identical data can land in byte-different files.

| Schema | Shard File | File SHA-256 | Content SHA-256 |
|---|---|---|---|
| **S0** | `playsite-bots-sample-000.parquet` | `129c834988b1840c...` | `be34d160e3c48bcd...` |
| **S0** | `playsite-bots-sample-001.parquet` | `3d87da6f5e4323e6...` | `0548343bd122b234...` |
| **S1** | `playsite-bots-sample-000.parquet` | `301e2a7d19025d2d...` | `fceec0a5cf9b5ae7...` |
| **S1** | `playsite-bots-sample-001.parquet` | `0015204198be9fd1...` | `c5513165aabae2dc...` |
| **S2** | `playsite-bots-sample-000.parquet` | `e660ad1dd55f43f3...` | `91fb838039a4078a...` |
| **S2** | `playsite-bots-sample-001.parquet` | `31eaae689411dd4e...` | `f1657c7064fffa3f...` |

## 2. Primary Estimand: Single-Model Replication

Evaluation of single-model performance across independent random seeds (mean ± std). Paired 95% confidence intervals are computed via whole-game bootstrap resampling on the mean seed loss delta of the primary estimand:

| Schema | Features | Log Loss | 95% CI vs S0 (Δ) | Brier | ECE | Rel. LL Gain | Gate Status |
|---|---|---|---|---|---|---|---|
| **S0** (`kcp-13`) | 13 | 0.6385 ± 0.0038 | — (Reference) | 0.2221 ± 0.0009 | 0.0350 ± 0.0061 | — | Baseline |
| **S1** (`kcp-mobility-27-v1`) | 27 | 0.6497 ± 0.0063 | [-0.0025, +0.0303] | 0.2239 ± 0.0015 | 0.0408 ± 0.0076 | -1.75% | FAILED |
| **S2** (`kcp-mobility-pawns-31-v1`) | 31 | 0.6443 ± 0.0039 | [-0.0014, +0.0138] | 0.2233 ± 0.0008 | 0.0391 ± 0.0094 | -0.90% | FAILED |

## 3. Unseen Validation Positions Performance

Evaluation on validation positions with no FEN/side overlap in the training partition:

| Schema | Unseen Positions | Log Loss | 95% CI vs S0 (Δ) | Brier | ECE | Unseen Gate |
|---|---|---|---|---|---|---|
| **S0** (`kcp-13`) | 3,805 | 0.6303 ± 0.0046 | — (Reference) | 0.2178 ± 0.0011 | 0.0381 ± 0.0086 | Baseline |
| **S1** (`kcp-mobility-27-v1`) | 3,805 | 0.6443 ± 0.0076 | [-0.0023, +0.0387] | 0.2202 ± 0.0017 | 0.0478 ± 0.0079 | FAIL |
| **S2** (`kcp-mobility-pawns-31-v1`) | 3,805 | 0.6376 ± 0.0046 | [-0.0013, +0.0164] | 0.2194 ± 0.0009 | 0.0440 ± 0.0092 | FAIL |

### 2b. Secondary Diagnostic: 5-Model Ensemble

> [!NOTE]
> Ensemble predictions (average probability across 5 seeds). Reported for variance-reduction diagnostics; not the single-model deployable contract.

| Schema | Features | Ensemble Log Loss | Ensemble Brier | Ensemble ECE |
|---|---|---|---|---|
| **S0** (`kcp-13`) | 13 | 0.6359 | 0.2213 | 0.0279 |
| **S1** (`kcp-mobility-27-v1`) | 27 | 0.6448 | 0.2224 | 0.0334 |
| **S2** (`kcp-mobility-pawns-31-v1`) | 31 | 0.6407 | 0.2220 | 0.0347 |

### Gate Checklist

| Gate Rule | S1 (`kcp-mobility-27-v1`) | S2 (`kcp-mobility-pawns-31-v1`) | Requirement |
|---|---|---|---|
| `relative_log_loss_gain_gte_1pct` | **FAIL** | **FAIL** | >= +1.0% relative gain on full validation |
| `paired_ci_upper_lt_0` | **FAIL** | **FAIL** | Paired 95% group-bootstrap CI upper bound < 0 on mean seed delta |
| `brier_no_regression` | **FAIL** | **FAIL** | Brier regression <= 0.0000 |
| `ece_regression_lte_0_01` | **PASS** | **PASS** | ECE regression <= 0.0100 |
| `slice_log_loss_lte_0_01` | **FAIL** | **FAIL** | Max slice log loss regression <= 0.0100 |
| `slice_brier_lte_0_01` | **FAIL** | **PASS** | Max slice Brier regression <= 0.0100 |
| `unseen_positions_exist` | **PASS** | **PASS** | Non-empty unseen validation partition (no-position-leakage) |
| `unseen_log_loss_lte_0_01` | **FAIL** | **PASS** | Unseen positions log loss regression <= 0.0100 |
| `unseen_brier_no_regression` | **FAIL** | **FAIL** | Unseen positions Brier regression <= 0.0000 |
| `extraction_cost_evidence_valid` | **PASS** | **PASS** | Verified JVM extraction benchmark artifact present |
| `extraction_cost_lte_tolerance` | **PASS** | **PASS** | Mean JVM extraction latency overhead <= 10.0% relative to S0 |
| `beats_no_information` | **PASS** | **PASS** | Better than the no-information reference on the primary metric |

## 4. Predeclared Slices Performance (Single-Model Means)

| Slice | S0 Log Loss | S1 Log Loss (Δ) | S2 Log Loss (Δ) | S0 Brier | S1 Brier (Δ) | S2 Brier (Δ) |
|---|---|---|---|---|---|---|
| `phase:opening` | 0.6603 | 0.6649 (+0.0046) | 0.6654 (+0.0050) | 0.2342 | 0.2359 (+0.0017) | 0.2361 (+0.0019) |
| `phase:middlegame` | 0.5857 | 0.5876 (+0.0019) | 0.5882 (+0.0025) | 0.2002 | 0.2004 (+0.0001) | 0.2005 (+0.0002) |
| `phase:endgame` | 0.7767 | 0.8960 (+0.1193) | 0.8079 (+0.0312) | 0.2551 | 0.2679 (+0.0128) | 0.2558 (+0.0007) |
| `side:w` | 0.6352 | 0.6508 (+0.0156) | 0.6409 (+0.0057) | 0.2214 | 0.2246 (+0.0032) | 0.2229 (+0.0014) |
| `side:b` | 0.6421 | 0.6486 (+0.0065) | 0.6479 (+0.0058) | 0.2228 | 0.2232 (+0.0004) | 0.2237 (+0.0009) |
| `material:behind` | 0.6327 | 0.6426 (+0.0098) | 0.6445 (+0.0117) | 0.2171 | 0.2178 (+0.0007) | 0.2194 (+0.0023) |
| `material:balanced` | 0.6643 | 0.6683 (+0.0040) | 0.6658 (+0.0015) | 0.2363 | 0.2381 (+0.0017) | 0.2371 (+0.0008) |
| `material:ahead` | 0.6031 | 0.6280 (+0.0250) | 0.6081 (+0.0051) | 0.2049 | 0.2085 (+0.0036) | 0.2052 (+0.0003) |
| `tactical:king_attack` | 0.6330 | 0.6401 (+0.0072) | 0.6388 (+0.0058) | 0.2202 | 0.2213 (+0.0012) | 0.2218 (+0.0017) |
| `tactical:king_danger` | 0.6380 | 0.6458 (+0.0078) | 0.6435 (+0.0054) | 0.2222 | 0.2241 (+0.0019) | 0.2236 (+0.0013) |
| `tactical:queen_attack` | 0.6376 | 0.6394 (+0.0018) | 0.6404 (+0.0028) | 0.2235 | 0.2242 (+0.0006) | 0.2243 (+0.0007) |
| `tactical:queen_danger` | 0.6374 | 0.6413 (+0.0039) | 0.6399 (+0.0025) | 0.2238 | 0.2250 (+0.0011) | 0.2245 (+0.0007) |

## 5. Calibration Analysis (Ensemble Diagnostic)

| Bin Range | S0 Count | S0 Pred / Obs | S1 Count | S1 Pred / Obs | S2 Count | S2 Pred / Obs |
|---|---|---|---|---|---|---|
| [0.0, 0.1) | 124 | 0.057 / 0.258 | 130 | 0.047 / 0.238 | 149 | 0.053 / 0.262 |
| [0.1, 0.2) | 206 | 0.153 / 0.214 | 190 | 0.150 / 0.189 | 243 | 0.154 / 0.202 |
| [0.2, 0.3) | 373 | 0.256 / 0.260 | 395 | 0.257 / 0.276 | 473 | 0.256 / 0.292 |
| [0.3, 0.4) | 677 | 0.350 / 0.362 | 623 | 0.351 / 0.369 | 642 | 0.353 / 0.386 |
| [0.4, 0.5) | 628 | 0.450 / 0.490 | 635 | 0.453 / 0.461 | 702 | 0.453 / 0.487 |
| [0.5, 0.6) | 1,159 | 0.539 / 0.525 | 1,093 | 0.541 / 0.524 | 1,075 | 0.533 / 0.549 |
| [0.6, 0.7) | 566 | 0.650 / 0.636 | 547 | 0.648 / 0.618 | 520 | 0.651 / 0.658 |
| [0.7, 0.8) | 431 | 0.746 / 0.735 | 445 | 0.749 / 0.710 | 379 | 0.746 / 0.723 |
| [0.8, 0.9) | 330 | 0.845 / 0.797 | 324 | 0.848 / 0.809 | 294 | 0.849 / 0.820 |
| [0.9, 1.0) | 134 | 0.941 / 0.843 | 246 | 0.945 / 0.813 | 151 | 0.944 / 0.828 |

## 6. Probe Suite Behavior

| Check | S0 | S1 | S2 |
|---|---|---|---|
| `canonical:ep-e6==ep-none` | PASS | PASS | PASS |
| `endgame:kings-only-finite` | PASS | PASS | PASS |
| `material:knight-up>start-w>knight-down` | FAIL | PASS | PASS |
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

## 7b. What This Corpus Can Settle

The public sample decides which schema the development protocol selects. It cannot settle whether the wider schemas help on the private corpus, and a failed gate here is not evidence that they do not:

- **S1** (`kcp-mobility-27-v1`, 27 features): paired mean-seed log-loss delta against S0 is [-0.0025, +0.0303]. The interval spans zero, so this corpus separates the two schemas from each other no better than it separates them from noise.
- **S2** (`kcp-mobility-pawns-31-v1`, 31 features): paired mean-seed log-loss delta against S0 is [-0.0014, +0.0138]. The interval spans zero, so this corpus separates the two schemas from each other no better than it separates them from noise.

The training partition holds 39,121 rows. Fitting 27 to 31 inputs on that many rows without regularisation is data-limited, and the owner's evidence for the wider schemas came from a corpus orders of magnitude larger. Read this report as: the development protocol selects S0 and the wider schemas cost nothing measurable in extraction latency, while the question the schemas were proposed to answer stays open for the private qualification run.

## 8. Next Actions

1. **Private Qualification**: Issue #17 remains open pending owner execution of the frozen protocol on the private corpus.
2. **ADR 0001 Maintenance**: Retain provisional development findings in ADR 0001; final amendment occurs after owner qualification.
3. **Value Model Training**: S0 remains the current baseline contract for #13.
