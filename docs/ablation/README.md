# Playground Feature Schema Ablation Protocol

Status: **AMENDED (Protocol v4)**. The default is `docs/ablation/protocol-v4.json`.
Historical `protocol-v1.json`, `protocol-v2.json` and `protocol-v3.json` remain
unchanged and can be selected explicitly with `--protocol`.

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
- **Protocol v3 Amendment (`playground-feature-ablation-v3`)**: Trains on logits with `BCEWithLogitsLoss` so a saturated sigmoid cannot block the corrective gradient; features, capacity and evaluation are unchanged.
- **Protocol v4 Amendment (`playground-feature-ablation-v4`)**: The validity amendment of
  [Issue #27](https://github.com/fortemate/dicechess-training/issues/27). Adds an admissibility
  floor against the no-information reference, standardises features on training statistics, and
  selects the epoch budget on an inner tuning split. Schemas, split policy, seeds, metrics,
  slices, probes, uncertainty and every gate threshold are inherited from v3 unchanged.

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

## Admissibility floor (Protocol v4)

The gate of protocol v2 compares the candidate schemas with S0 and with nothing else. A run whose
models are all worse than predicting the base rate therefore still produced a confident winner:
that is what the protocol-v2 report did, with every arm between 0.80 and 1.13 validation log loss
against 0.6926 for the constant predictor
([evidence](https://github.com/fortemate/dicechess-training/issues/17#issuecomment-5670027600)).

Protocol v4 scores the reference the benchmark of Issue #13 already uses — a constant equal to the
training base rate — and records it in both report formats. A schema whose mean single model does
not beat it is **inadmissible**: its gate fails, and it cannot be selected. When even S0 is
inadmissible the run selects nothing, the decision records `inadmissible-no-selection`, and the CLI
writes its reports and then exits non-zero, so a pipeline cannot read "no selection" as a pass.

The floor is a validity check on each arm on its own. It does not touch the relative
S1/S2-versus-S0 thresholds frozen in v2.

## Feature standardisation and the serving contract (Protocol v4)

Protocol v4 fits mean and scale on the **training partition only** and carries them inside the
model as persistent buffers applied before the first layer. Three consequences matter:

- the model consumes **raw** schema features, so `input [batch, N]` -> `output [batch, 1]` from
  ADR 0001 is unchanged and the evaluator needs no new step;
- the scaler travels with the checkpoint and with the exported graph
  (`dicechess_training.ablation.export.export_candidate`), so a candidate cannot be served
  without it. `export.probe_parity` checks the exported graph against the Python pipeline on the
  engine's golden probes, and the contract validator checks names, shapes, dtype and opset;
- a constant training column keeps a scale of 1.0, so a feature that never varies is centred
  rather than amplified into noise.

Protocols without `model.feature_standardisation` train on raw columns exactly as before and
register no buffers, which keeps their checkpoint layout byte-identical.

## Epoch budget (Protocol v4)

Protocols v1-v3 fix five epochs. Under v4 the budget is chosen per seed from
`model.epoch_selection.candidates` by log loss on an **inner tuning split** carved out of the
training partition (`inner_cutoff` on the same deterministic game hash), after which the model is
refitted on the whole training partition with the chosen count. The validation partition is never
used to choose the budget: it stays a reporting and gating surface. One training pass serves all
candidates by scoring at each candidate epoch, so selection costs one extra fit per seed rather
than one per candidate.

## Numerically stable training (Protocol v3)

Protocol v3 amends v2 by setting `model.loss` to `bce-with-logits`.
The trainer passes unbounded logits directly to `BCEWithLogitsLoss`, while
`ValueMLP.forward` and `predict` continue returning probabilities. Checkpoint
parameter names and the sigmoid used for inference are preserved.

With the historical `Sigmoid` followed by `BCELoss`, float32 sigmoid can round
a confidently wrong prediction to exactly zero or one. The saturated sigmoid
then blocks the corrective gradient. More epochs do not address that numerical
failure. The fused logits loss retains the gradient without altering feature
definitions or normalizing inputs.

Protocols without `model.loss` retain historical `bce` behavior. Explicit `bce`
also selects that path; unsupported loss names fail closed. The amendment leaves
architecture, optimizer, seeds, epoch budget, split, metric clipping and gates
unchanged. Reports carry the actual protocol hash: never pool v2 and v3 runs or
replace historical evidence. Raw feature scaling remains a separate experiment.
The probability-based log loss still clips at the documented epsilon; it must
not be changed to conceal saturated errors.

Regression coverage exercises both extreme-logit gradient directions, actual
learning from a saturated initialization, historical behavior and checkpoint
compatibility. See [issue 26](https://github.com/fortemate/dicechess-training/issues/26).

## Decision Gate (unchanged from Protocol v2)

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

Shard provenance is recorded twice. The **file digest** identifies the artifact a run read. The
**content digest** (`dicechess_training.schema.shard_content_digest`) covers the semantic metadata
and the column values in a canonical order and is what an independent reproduction compares: the
Parquet writer records its per-column encoding set in the footer in an order that varies between
producer sessions, so two byte-different files can hold identical rows. Reproducing a report is
expected to match the content digests, not the file digests.

Extraction-cost evidence must contain the complete probe set from the golden
corpus for the protocol's engine version, with matching FENs and candidate schema
identities. Each latency statistic must be numeric, finite and positive, with
`min <= median <= p95 <= max`. Engine provenance, runtime fields and positive
warmup/sample counts are required. Both report loading and gate evaluation reject
partial or incompatible evidence. The mean relative overhead uses compensated
summation so a candidate exactly at the declared ceiling is not rejected due to
accumulated rounding error. These validations do not change the protocol v2
selection threshold or turn development measurements into final qualification.

### Reproducing the public-sample report

The runner reads one directory per schema, each named by its **schema identifier**, so the
enrichment step writes `<data-dir>/<schema-id>/`:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 21)   # Hadoop rejects newer JDKs
cd tools/kcp13-golden
for schema in kcp-13 kcp-mobility-27-v1 kcp-mobility-pawns-31-v1; do
  sbt -batch -Dengine.version=0.9.3 \
    "runMain dicechess.training.golden.EnrichShardsApp ../../sample/playsite-bots-v0 ../../data/enriched/$schema $schema"
done
cd ../..
uv run python -m dicechess_training.ablation --data-dir data/enriched --output-dir out/ablation
```

A reproduction is expected to match the report's content digests, its metrics and its decision;
the file digests belong to the producer run that made them.

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
