# Playground Feature Schema Ablation Protocol

Status: **AMENDED (Protocol v2)**. Protocol v2 (`docs/ablation/protocol-v2.json`) supersedes historical Protocol v1 (`docs/ablation/protocol-v1.json`).

## Context & Protocol Lineage

Parent epic: [Issue #12](https://github.com/fortemate/dicechess-training/issues/12).
Decision record: [ADR 0001](../decisions/0001-playground-train-serve-contract.md).
Benchmark specification: [Benchmark v1](../benchmark/README.md).

The evaluation service serves `standard-kcp` over the `kcp-13` feature schema. Owner research indicates that position properties such as expected wasted rolls, pawn blockage, tempo as independent own/opponent mobility, and passed pawns are critical for one-ply position evaluation. The engine implements these features as versioned extractors (`kcp-mobility-27-v1` and `kcp-mobility-pawns-31-v1`) in `com.fortemate:dicechess-engine_3:0.9.3` (engine Issue #215).

### Protocol Version History

- **Protocol v1 (`playground-feature-ablation-v1`)**: Initial historical pre-results protocol definition.
- **Protocol v2 Amendment (`playground-feature-ablation-v2`)**: Supersedes v1 with four key methodological enhancements:
  1. **Split Boundary Isolation**: Enforces an 80/10/10 game split policy (`train < 8000`, `val [8000, 9000)`, `test >= 9000`), strictly excluding test holdout rows from candidate selection.
  2. **Primary Estimand Bootstrap**: Formally evaluates single-model replication across 5 predeclared random seeds (`[11, 23, 47, 89, 131]`) and computes paired 95% group-bootstrap confidence intervals directly on the mean seed loss delta for each resampled whole-game draw.
  3. **Unseen-Position Gating**: Requires non-empty unseen validation positions and gates against regressions on unseen positions (`position_key` not seen in train).
  4. **Extraction Cost Enforcement**: Integrates verified JVM benchmark evidence into the candidate selection gate (mean probe latency overhead <= 10.0% relative to S0; missing or over-budget evidence blocks qualification).

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
  - Validation: $8000 \le \text{value} < 9000$ (10% of games, used for candidate selection)
  - Test holdout: value $\ge 9000$ (10% of games, reserved holdout strictly excluded from schema selection)
- **Position leakage audit**: Exact canonical position overlap (`position_key`, FEN and side-to-move) is audited across splits (`train:validation`, `train:test`, `validation:test`).

## Evaluation & Estimand

- **Primary Estimand**: Single-model replication across 5 predeclared random seeds (`[11, 23, 47, 89, 131]`). Reports per-metric mean and standard deviation across seeds. Selection gates and paired confidence intervals evaluate this single-model distribution because the evaluation service serves a single model instance.
- **Secondary Diagnostic**: A 5-model ensemble ($p_{\text{ens}} = \frac{1}{5}\sum_k p_k$) is evaluated separately for variance-reduction diagnostics and is clearly labeled as non-serving.
- **Primary metric**: Log-loss $-\frac{1}{N}\sum [y \ln(p) + (1-y) \ln(1-p)]$, clipped at $10^{-15}$.
- **Calibration metrics**: Brier score $\frac{1}{N}\sum (p - y)^2$ and Expected Calibration Error (ECE) across 10 equal-width bins.
- **Uncertainty**: 95% bootstrap confidence intervals computed by resampling whole games/groups (1,000 resamples, seed 13) on the mean seed delta.
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
- **Feature Extraction Latency**: Evaluated via verified JVM benchmark tool (`ExtractionBenchmarkApp`) capturing runtime, OS, and JVM engine provenance (`tests/fixtures/benchmark/extraction-cost-0.9.3.json`).

## Decision Gate (Protocol v2)

A wider candidate (S1 or S2) is selected over S0 iff:
1. Relative log-loss improvement on validation: $\frac{\text{LL}_{S0} - \text{LL}_S}{\text{LL}_{S0}} \ge 1.0\%$.
2. Paired 95% whole-game bootstrap CI upper bound of mean seed log-loss delta: $\Delta \bar{\text{LL}}_{97.5} < 0$.
3. Brier score does not regress: $\text{Brier}_S - \text{Brier}_{S0} \le 0.0$.
4. ECE does not regress beyond tolerance: $\text{ECE}_S - \text{ECE}_{S0} \le 0.01$.
5. Critical slices do not regress beyond point-estimate tolerance: $\Delta \text{LL}_{\text{slice}} \le 0.01$, $\Delta \text{Brier}_{\text{slice}} \le 0.01$.
6. Unseen positions partition is non-empty and does not regress: $\Delta \text{LL}_{\text{unseen}} \le 0.01$, $\Delta \text{Brier}_{\text{unseen}} \le 0.0$.
7. Serving extraction cost measured on JVM is verified and bounded (mean probe latency overhead <= 10% relative to S0; missing or over-budget evidence blocks qualification).

## Qualification & Publication Boundary

- **Provisional Development Evidence**: Runs executed on `sample/playsite-bots-v0` verify tooling, pipeline execution, and protocol mechanics. Because the public sample previously informed earlier review steps under Protocol v1, public sample runs under Protocol v2 are provisional development evidence and are ineligible for final qualification without reviewed data-use evidence.
- **Owner Qualification**: Final schema qualification is run by the repository owner on an eligible holdout that has not previously informed selection under [Issue #17](https://github.com/fortemate/dicechess-training/issues/17).
- **Private Decision Reference**: The definitive qualification decision is recorded in the private knowledge base under page title:
  `Private Decision: Playground Feature Schema Qualification (Issue #17)`.
  Per publication boundaries, private numerical metrics and final qualification outcomes are preserved in private documentation and not published in public Git.


## Input validation and reproducible checks

Enriched shards must declare `ruleset=standard-dicechess-v1` and
`perspective=side-to-move`, in addition to the expected feature schema and engine
version. The JVM producer writes these fields and rejects a source `side` that
contradicts the FEN active color. The Python reader rejects missing or incompatible
semantic metadata before exposing features or labels.

Older enriched shards without these fields must be regenerated from the audited
schema-v0 source using `EnrichShardsApp`; do not attach labels to unverified legacy
features merely to bypass validation. Existing historical reports retain their
original input digests. New runs must record the newly generated shard digests and
write results to an ignored/private output destination.

Extraction-cost evidence must contain the complete probe set from the golden
corpus for the protocol's engine version, with matching FENs and candidate schema
identities. Each latency statistic must be numeric, finite and positive, with
`min <= median <= p95 <= max`. Engine provenance, runtime fields and positive
warmup/sample counts are required. Both report loading and gate evaluation reject
partial or incompatible evidence. The mean relative overhead uses compensated
summation so a candidate exactly at the declared ceiling is not rejected due to
accumulated rounding error. These validations do not change the protocol v2
selection threshold or turn development measurements into final qualification.

Run `mise run check` for Python checks and `mise run check:enrichment` for the
JVM → Parquet → Python contract smoke. The latter requires **JDK 21** and sbt;
it verifies all three feature schemas against engine golden vectors and confirms
that invalid source perspective does not publish a partial shard. CI runs this
smoke separately on JDK 21. The current Hadoop dependency is incompatible with
newer JDKs (reproduced on JDK 26) (`Subject.getSubject`); set `JAVA_HOME` to JDK 21 for enrichment.

CLI input paths (`--data-dir`, `--protocol`, `--extraction-cost`) are read-only:
existing directories/files may be on a private mount outside the repository.
Output paths retain their separate location checks and exclusive creation policy.
Generated reports identify their actual inputs through shard digests rather than
assuming that every run uses the public sample. Every selection, including S1/S2
success, remains provisional until separate owner qualification.

### Parallel shard enrichment

The JVM producer evaluates each bounded chunk using an ordered parallel-stream
collection before writing records sequentially. Calling `iterator()` directly
on the mapped parallel stream would instead evaluate the mapping sequentially.
Only one chunk of feature vectors is materialized at a time, and source row order
and atomic shard publication are preserved. Extraction failures still reject the shard.

`mise run check:enrichment` runs the JVM concurrency/order regression tests before
the JVM-to-Python golden and rejection checks. The concurrency test uses its own
bounded pool and requires overlapping extractor calls; it does not rely on a
wall-clock speed threshold or the host's default processor count.
