# ADR 0001: Train/serve contract for the first real playground evaluation model

- Status: Accepted
- Date: 2026-09-11
- Decision owner: repository maintainer
- Tracking: [Issue #14](https://github.com/fortemate/dicechess-training/issues/14)
  (parent epic [#12](https://github.com/fortemate/dicechess-training/issues/12))

## Context

The protected evaluation playground is deployed and its browser → Cloudflare Access → BFF →
private evaluator path is verified end to end, but the model it serves is a synthetic fixture
whose probabilities carry no Dice Chess meaning. The private evaluation service accepts exactly
one contract, `standard-kcp` over the `kcp-13` feature schema, while this repository's roadmap
targets richer contracts for different **model roles**: a raw-board/NNUE chance-collapse afterstate
network ([#8](https://github.com/fortemate/dicechess-training/issues/8)) and a listwise move
pre-ranker ([#9](https://github.com/fortemate/dicechess-training/issues/9)). A model can have good
offline metrics and still be unusable or semantically wrong if training features, serving
features, probability perspective or role differ.

This record decides the first candidate's train/serve contract, proves the selected semantics
with a shared golden corpus, and inventories what exists. It is an engineering decision by the
maintainer; the benchmark and promotion gate are decided separately in
[#13](https://github.com/fortemate/dicechess-training/issues/13).

## Inventory: the contract the evaluator serves today

Read from the private `dicechess-evaluation` main branch and the public engine on 2026-09-11. Only the
public train/serve contract is recorded here; the service's operational configuration (limits,
caching, deployment) stays in its private documentation.

| Item | Value |
| --- | --- |
| Evaluation profile | `standard-kcp` (the only supported id; unknown ids are rejected) |
| Algorithm | `kcp-1ply-onnx`: one-ply candidate analysis, transposition-deduplicated, scored by the ONNX model |
| Feature schema | `kcp-13`, exactly 13 FLOAT features |
| Feature source | engine `KcpFeatures.extract(state, state.activeColor)` — `RichFeatures` (7 material + `mobility_diff` + `king_safety_diff`) followed by `king_capture_attack`, `king_capture_danger`, `queen_capture_attack`, `queen_capture_danger` |
| Column order | pinned in `dicechess_training.contracts.kcp13.COLUMN_NAMES` and in the golden corpus |
| Perspective | **the side to move** (own minus opponent); the vector is mover-canonical, so a position and its colour-swapped rank-mirror encode identically |
| Dice | **none** — every column is dice-independent; the request for a position evaluation carries no dice |
| ONNX input | exactly one FLOAT tensor named `input`, shape `[batch, 13]` with a dynamic batch axis |
| ONNX output | exactly one FLOAT tensor named `output`, shape `[batch, 1]` with a dynamic batch axis |
| Output handling | one finite value per row, clamped to `[0, 1]`; `NaN`, infinities and row-count mismatches fail the inference |
| Output meaning | `POST /api/v1/evaluate/position`: P(the side to move wins). `POST /api/v1/evaluate/turn`: each candidate afterstate is scored from its *new* side to move and reported as `1 − p` for the mover; an immediate king capture is reported as `1.0` without a model call |
| Manifest | `manifestVersion` `1.0.0`; `modelId` non-blank; `modelSha256` = 64 hex of the ONNX bytes; `featureSchema` `kcp-13`; `featureCount` 13; `evaluationProfile` `standard-kcp`; `engineCompatibility` = space-separated `MAJOR.MINOR.PATCH` comparators (`=`, `>`, `>=`, `<`, `<=`, AND semantics, bare version = equality) that must include the serving engine version; optional `calibration` and free-form `provenance` |
| Serving engine | the public `com.fortemate:dicechess-engine_3` artifact, version 0.9.2 at the time of writing; the version is part of every response's provenance |
| Runtime | ONNX Runtime on the JVM; the qualification step verifies that the exported opset loads there |
| Fail-closed points | before serving: manifest validation, model digest, ONNX graph contract; per request: feature width, output shape and finiteness |

### Feature cost and serving envelope

Extraction, not inference, is the cost of this contract. Measured with the golden generator
(`tools/kcp13-golden`, engine 0.9.2, JIT-warm, single thread, laptop-class CPU) the 13-feature
vector takes **0.03–4.4 ms per position** depending on how many pieces the 216-outcome
capture-probability search has to expand: bare kings 0.13 ms, the initial position ≈ 2 ms, an open
middlegame ≈ 3.6–3.8 ms, the position after `1.e4 e5` ≈ 4.4 ms. That agrees with the evaluator's
own documentation (≈ 5 ms per candidate on the production host) and dominates a turn analysis,
where the opening position with dice `PPN` already expands 2 540 paths into 1 028 distinct
candidate positions. Any model on this contract therefore has to add only negligible batched
inference time (well under 0.1 ms per row); the envelope is set by feature extraction, which is
already in place and unchanged by this decision.

### Feature semantics across engine releases

`KcpFeatures`, `RichFeatures` and `OnnxFeatures` have not changed since engine `v0.2.0`;
`KingCaptureProbability` was refactored in `v0.8.0` (engine #118). The golden corpus below was
replayed against the released engines **0.4.0, 0.7.2 and 0.9.2** and produced byte-identical
vectors for all 23 probes. The private analytics enrichment pins engine 0.9.1 and the evaluator
0.9.2, both inside that range.

Re-verified on 2026-09-17 under [Issue #30](https://github.com/fortemate/dicechess-training/issues/30):
the corpus, now 27 probes, replays byte-identically on **0.9.3** and on **0.12.0**, the release line
that carried the open-core split of engine ADR 009. That split moved `FenParser`, `GameState` and
`TurnGenerator` into `com.fortemate:dicechess-rules_3`, while `KcpFeatures` stayed in
`com.fortemate:dicechess-engine_3` because it reaches `RichFeatures` and through it the evaluator — so
a consumer of the feature contract still depends on the engine artifact, not on the rules artifact
alone. `tests/test_kcp13_contract.py` asserts the agreement between every committed golden instead of
recording it in prose.

## Inventory: model roles

| Role | Input object | Perspective | Dice | Target | Servable through `/evaluate/position` today? |
| --- | --- | --- | --- | --- | --- |
| **Position evaluation** (this decision) | a position with its side to move, before that side rolls | side to move | unknown | P(side to move wins) | yes — this is the contract |
| Chance-collapse afterstate value (#8) | the same physical object: a position after a move and before the next roll | the **previous** mover, i.e. `1 − p` of the side to move | unknown | depth-3 teacher expectation, not an outcome probability | **no** — needs a perspective adapter, a successor feature schema and a benchmark whose target is the teacher value; equal tensor width would prove nothing |
| Listwise move pre-ranker (#9) | a root position **with the current dice** and its whole legal candidate list | mover | known | exact rescore distribution over candidates | **no** — no per-position probability exists; the role is list scoring |

The evaluator's turn analysis already performs the `1 − p` flip for candidate afterstates, so a
future afterstate model would be an *implementation* of the position role only if its output were
a calibrated win probability under the same feature schema — a benchmark question (#13), not a
tensor question.

## Inventory: historical candidate families

Sanitised. Private artifacts are identified by schema and SHA-256 digest only; no storage
location, bot configuration or weight is disclosed.

| Family | Width | Export / graph | Meets the tensor contract? | Lineage |
| --- | --- | --- | --- | --- |
| material GBDT (`OnnxFeatures`) | 7 | OnnxMLTools `TreeEnsembleRegressor`, output tensor `variable` | no (width, output name) | private arena logs only |
| rich GBDT (`RichFeatures`) | 9 | same | no (width, output name) | private arena logs only |
| **kcp GBDT (`KcpFeatures`)**, one artifact, SHA-256 `df6f5252ad31…` (private) | 13 | OnnxMLTools `TreeEnsembleRegressor`, LightGBM `regression` objective, output tensor `variable` | **no**: output must be named `output`; regression output is unbounded and would rely on the evaluator's clamp | training export, enrichment engine version, config and commit are **not recorded next to the artifact** → recorded as a blocker; it cannot become the promotion reference |
| safe GBDT | 15 | same | no (width) | private |
| raw-board MLP (`RawBoardFeatures`) | 768 | PyTorch, `input`/`output` | no (schema) | private; measured −25 Elo against the rich GBDT at two plies |
| public demo value net (`mise run demo`) | 774 = raw board + dice | PyTorch, `features`/`win_probability` | no (schema, names) and **wrong role**: dice-conditioned | this repository |

The private `dicechess-ev` pipeline already trains a dice-free `kcp` LightGBM (`--kcp --no-dice`)
from a CSV enriched by the analytics `EnrichTrainingDataApp`, which calls the very same
`KcpFeatures.extract`. That confirms the train == serve principle for this schema but does not
make any existing artifact deployable: none of them was exported against the manifest contract,
and their datasets and configs are not bound to them immutably.

## Feature-schema ablation

Choosing the schema is a measured decision, not a default. The owner's evidence says features,
not data volume, bound quality (a 17 % larger corpus moved AUC by −0.002; the four
capture-probability columns added +2.2 pp AUC on 17 M rows), and the owner's strategy research
(September 2026) names position properties that no served schema exposes: **expected wasted
rolls** (a die face whose piece type has no move), **pawn blockage**, **tempo** as separate
own/opponent mobility, and **passed pawns** whose value depends on rolling the right die. All of
them derive from pseudo-legal move counts *per piece type, per side*, which `RichFeatures`
already computes and collapses into one `mobility_diff`, so their marginal extraction cost is
small next to the 216-outcome capture-probability search. The engine exposes them as versioned
extractors in [engine #215](https://github.com/fortemate/dicechess-engine/issues/215);
[Issue #17](https://github.com/fortemate/dicechess-training/issues/17) runs the ablation.

| Schema | Columns | Contents |
| --- | ---: | --- |
| **S0** `kcp-13` | 13 | the served baseline |
| **S1** `kcp-mobility-27-v1` | 27 | S0, then pseudo-legal move counts per piece type for the side to move and for the opponent (6 + 6), then own and opponent PDI (`PieceDiversity.count / 5`) |
| **S2** `kcp-mobility-pawns-31-v1` | 31 | S1, then passed-pawn count and most advanced passed-pawn rank for both sides |

Rules of the ablation: the protocol (model family, seeds, splits, metrics, slices, probe suite,
decision rule) is committed before the first result; every schema's result is reported,
including losers; the decision is relative to S0 under the #13 gate; a schema enters the serving
contract only through that gate and only as an engine-implemented extractor (train == serve).
The golden corpus gains engine-generated vectors for S1 and S2 with the same invariants as
`kcp-13` plus prefix identity and `sum(own_moves) − sum(opp_moves) == mobility_diff`. The
representation question raised during review (signed differences versus separate own/opponent
counts) is answered by S1's design: new blocks are own/opponent, and the ablation measures it.

Rejected for this ablation: a safety block (hanging material weighted by capture probability)
cured the queen-hang blind spot in the private `safe_1m` retrain but tripled extraction time and
has no public engine extractor; it remains a documented later candidate, not a schema here.

## Alternatives considered

### Reuse the existing 13-wide GBDT artifact

Rejected. Its output tensor is named `variable`, its regression output is unbounded, and its
dataset/config/code mapping is not recorded. Re-exporting it would change the digest and still
leave the lineage unverifiable. The DoD forbids reconstructing provenance from a file name.

### Adopt a successor schema now (dice-free `rawboard-768` or an NNUE-style encoding)

Rejected for the *first* candidate. It requires a versioned change in the private evaluator
(schema negotiation, a second extractor, profile, manifest allow-list, cache namespace, fixtures,
backward compatibility) before any training can be qualified, and the available evidence does not
favour it at one ply: the engine's own measurements put the one-ply raw-board net at 57.3 % versus
60.8 % for the `kcp` set against the same baseline, and its per-evaluation cost about 5× higher.
The public demo also shows the plain raw-board MLP over-fitting the public sample. The evidence
that would reopen this alternative is listed under *Revisit*.

### Serve the #8 or #9 artifacts through the position API

Rejected. Different role, perspective and target (see the role inventory). An adapter is not a
contract; a benchmark redefinition would be required first.

### Train a new `kcp-13` candidate inside the unchanged contract

Accepted, with the constraints below.

## Decision

1. **Role and contract.** The first real playground model is a *position evaluation* model:
   its output is **P(the side to move wins)**, dice-free, on the evaluator's existing tensor and
   manifest contract. `kcp-13` is the **reference baseline** and the only schema the evaluator
   serves today. The first candidate's **feature schema is selected by the predeclared ablation**
   in [Issue #17](https://github.com/fortemate/dicechess-training/issues/17) (see *Feature-schema
   ablation* below): if a wider schema clears the #13 gate against `kcp-13`, it enters the
   evaluator as one additive, versioned schema through one separately reviewed
   `dicechess-evaluation` Issue; otherwise `kcp-13` ships unchanged. Every report, model card and
   manifest states the perspective explicitly.
2. **Single feature source.** Training features are produced by the engine's
   `KcpFeatures.extract(state, state.activeColor)` at a pinned released engine version — the same
   code the evaluator runs. Python never reimplements the positional or capture-probability
   columns; `dicechess_training.contracts.kcp13.material_block` recomputes only the seven material
   columns and only as a test oracle for column order. The training follow-up adds a JVM
   enrichment step over schema-v0 shards that stamps the engine version and feature schema into
   the enriched shards' metadata.
3. **Shared golden corpus.** `tests/fixtures/kcp13/probes.tsv` (23 probes: opening symmetry,
   4-/6-field and en-passant canonicalisation, material imbalance, king- and queen-capture
   threats with exact analytic expectations, endgames, two public-sample middlegames, and seven
   colour-swapped twins) and the JVM-generated `golden-engine-0.9.2.json` are the contract fixture
   for both sides — the corpus as of this record; see *Amendment: Engine range and corpus replay*
   for the 27-probe corpus and the goldens committed since. `tools/kcp13-golden` regenerates it from a released engine; the Python tests in
   `tests/test_kcp13_contract.py` verify layout, perspective invariants, canonicalisation and the
   analytic capture-probability values (`91/216` for a single-face capture, `16/216` for a
   two-queen-move capture). Adding a probe or changing the engine version regenerates the file;
   editing it by hand is not allowed.
4. **Tensor contract of the export.** The candidate is exported with input `input` `[batch, 13]`
   FLOAT and output `output` `[batch, 1]` FLOAT, dynamic batch axis, sigmoid inside the graph,
   any input standardisation folded into the graph as constants, default-domain ONNX opset ≤ 18
   (the ceiling `kcp13.validate_onnx_contract` enforces; the evaluator's runtime loads it). A
   small PyTorch MLP is the default family because it meets the contract
   natively and its Torch↔ONNX parity is exact to float32; a GBDT is admissible only if its export
   meets the same contract and parity, and is then a comparison, not the default.
5. **Manifest.** Built with `kcp13.build_manifest` from the exact ONNX bytes; `engineCompatibility`
   is `>=0.4.0 <0.10.0` (widened to `>=0.4.0 <0.13.0` by *Amendment: Engine range and corpus
   replay*), the range replayed byte-identically by the golden corpus. Widening the
   range requires replaying the corpus (and, for a data release, the enrichment differential over
   the public sample) on the new engine release first. `provenance` records at least the training
   data digest(s), the enrichment engine version, the training git commit, the config digest, the
   seed and the benchmark report id.
6. **Parity protocol, fail closed.** (a) Python: PyTorch versus onnxruntime on the golden matrix
   and on held-out rows, max abs diff ≤ 1e-6. (b) Cross-runtime: the evaluator's
   `/evaluate/position` on every golden FEN versus onnxruntime on the golden vectors, max abs diff
   ≤ 1e-5 (float32 extraction). (c) Any mismatch in feature order, schema id, feature count,
   tensor names/shapes/dtypes, manifest version, profile, engine range or digest refuses the
   artifact on both sides; `kcp13.validate_manifest`, `kcp13.validate_onnx_contract` and
   `kcp13.verify_model_digest` mirror the evaluator's checks so the refusal happens before an
   artifact is ever mounted.
7. **Serving envelope.** No evaluator configuration changes for the candidate; qualification
   verifies latency, memory, concurrency and probability bounds inside the existing Aurora limits
   (feature extraction remains the dominant cost, see the inventory).
8. **Revisit.** Two different doors, two different keys. An *additive* schema that keeps the
   `kcp-13` prefix (the ablation candidates below) needs only the #13 gate. A *replacement*
   representation (dice-free raw board, NNUE) is opened as a separate `dicechess-evaluation`
   Issue only when both hold: the depth-3 label factory (#7) delivers teacher labels, and such a
   candidate beats the accepted baseline on the frozen #13 benchmark by its predeclared gate.

## Consequences

- The evaluator, playground and Aurora runbook need no change for the first candidate; the only
  new serving-side work is a parity fixture that replays the golden probes (tracked as a follow-up
  in the evaluator repository, not as a contract change).
- The quality ceiling is bounded by 13 hand-crafted features and by game-outcome labels; the
  benchmark in #13 must therefore compare against the no-information predictor and report
  calibration rather than accuracy, and the promotion gate is relative, not absolute.
- Already created: [engine #215](https://github.com/fortemate/dicechess-engine/issues/215)
  (versioned S1/S2 extractors), [analytics #40](https://github.com/fortemate/dicechess-analytics/issues/40)
  (the private enrichment application learns both schemas) and
  [#17](https://github.com/fortemate/dicechess-training/issues/17) (the ablation, which also
  delivers engine-computed enrichment of schema-v0 shards with provenance). Created after the ablation selects the schema: reproducible training and export of
  the candidate with manifest and digests; qualification against #13 (offline report, parity,
  latency, playground probe); the evaluator golden-parity fixture, plus the evaluator schema
  Issue if S1 or S2 wins.
- Recorded blockers: the historical 13-wide artifact has no immutable lineage and is not a
  promotion reference; the private enrichment path lives in the analytics repository and is not
  reproducible from a public checkout, which is why (1) exists.

## Approval

The authorized maintainer approved this decision on 2026-09-11 in the
[Issue #14 decision comment](https://github.com/fortemate/dicechess-training/issues/14#issuecomment-5639730674). The evaluator, engine and analytics repositories share the
same maintainer, so cross-repository review is covered by that comment; the benchmark (#13) must
adopt exactly the perspective in Decision 1, and its gate decides the ablation in #17.

## Amendment: Feature Schema Selection (Issue #17, Provisional Status)

Under [Issue #17](https://github.com/fortemate/dicechess-training/issues/17), the three candidate feature schemas (S0 `kcp-13`, S1 `kcp-mobility-27-v1`, S2 `kcp-mobility-pawns-31-v1`) are evaluated under amended Protocol v2 in [`docs/ablation/protocol-v2.json`](../ablation/protocol-v2.json), which supersedes historical Protocol v1 [`docs/ablation/protocol-v1.json`](../ablation/protocol-v1.json).

### Status, Protocol v2 Lineage, and Data Eligibility

1. **Protocol v2 Amendment**: Amends Protocol v1 by:
   - Enforcing an 80/10/10 split policy with strictly isolated test holdout (`train < 8000`, `val [8000, 9000)`, `test >= 9000`).
   - Defining the primary estimand as single-model replication across 5 predeclared seeds with paired whole-game bootstrap confidence intervals computed on the mean seed loss delta.
   - Gating candidates on non-empty unseen validation positions without regression.
   - Requiring verified JVM extraction latency benchmark evidence within a 10% relative overhead ceiling.
2. **Provisional Development Evidence**: Runs performed in this repository against the public sample (`sample/playsite-bots-v0`) confirm pipeline tooling, cross-schema integrity validation, and Protocol v2 mechanics. Because the public sample previously informed earlier review iterations under Protocol v1, public sample results under Protocol v2 represent provisional development evidence.
3. **Benchmark Custody & Eligibility**: Under Benchmark v1, the public sample is ineligible for final qualification without separately reviewed data-use evidence.
4. **Owner Qualification**: Final schema selection requires owner-run qualification on an eligible private holdout corpus under [Issue #17](https://github.com/fortemate/dicechess-training/issues/17).
5. **Private Decision Reference**: The definitive qualification decision is recorded in the private knowledge base:
   - Page title: `Private Decision: Playground Feature Schema Qualification (Issue #17)`.
   - In accordance with repository publication boundaries, private numerical metrics and final qualification outcomes are preserved in private documentation and are not published in public Git.
6. **Active Baseline**: Baseline **S0 (`kcp-13`)** remains the active train-serve contract for ongoing development and tooling under [Issue #13](https://github.com/fortemate/dicechess-training/issues/13).
7. **Issue State**: [Issue #17](https://github.com/fortemate/dicechess-training/issues/17) remains open pending completion of private corpus qualification and data eligibility review.

## Amendment: Engine range and corpus replay (Issue #30, 2026-09-17)

1. **Corpus.** The shared fixture is `tests/fixtures/kcp13/probes.tsv` with **27 probes** — the 23 of
   Decision 3 plus `blocked-pawns-w`, `passed-pawn-w` and their colour-swapped twins, added for the
   feature ablation — and three JVM-generated goldens: `golden-engine-0.9.2.json` (23 probes, the
   engine the evaluator pins today), `golden-engine-0.9.3.json` (27 probes, the ablation and
   enrichment engine) and `golden-engine-0.12.0.json` (27 probes, the current engine release).
2. **Replay evidence.** `kcp-13` extraction is unchanged across the released range: every probe that
   two goldens share carries bit-identical vectors, and the 0.12.0 file differs from the 0.9.3 one
   only in the recorded version string. `tests/test_kcp13_contract.py` enforces this, so an engine
   whose extraction drifts fails the suite instead of quietly redefining the served contract.
3. **Engine range.** Decision 5's `engineCompatibility` for `kcp-13` widens from `>=0.4.0 <0.10.0` to
   **`>=0.4.0 <0.13.0`**: replayed on 0.4.0, 0.7.2, 0.9.2, 0.9.3 and 0.12.0, with the upper bound left
   exclusive at the first unreplayed minor. Widening it further keeps the Decision 5 rule — replay the
   corpus on the new release first.
4. **Order of operations.** `dicechess-evaluation` validates a mounted model package against its own
   engine version and refuses the package at startup when the manifest range excludes that version. A
   manifest issued with the narrower range must therefore be re-issued with the widened one **before**
   that service's engine pin moves, or the service starts without a model.
5. **Unchanged.** The ablation stays pinned to engine 0.9.3 (`docs/ablation/protocol-v4.json`,
   `scripts/check_enrichment.py`); nothing here changes the selected schema, the tensor contract, the
   manifest rules or the promotion gate.
