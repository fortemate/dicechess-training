# Playground benchmark v1

Status: proposed protocol for owner review in Issue #13. No trained candidate has been selected
or qualified by this change. Review and commit this protocol before accepting candidate training.
The machine-readable policy is [protocol-v1.json](protocol-v1.json); changing any rule starts a
new benchmark version and requires a new untouched final holdout. This is a qualification tool
for ADR 0001's `standard-kcp` position evaluator, not an arena or deployment command.

## Task and interpretation

The model receives the engine's dice-free, mover-canonical `kcp-13` vector and predicts a win
probability for the side to move. One observation is a recorded pre-turn position with the final
game outcome from that side's perspective. Only decisive games enter the binary scores: win = 1,
loss = 0. Draws (input result 0.5) are counted and excluded. Consequently these metrics measure
**win prediction conditional on decisive games**, not unconditional win probability, expected
score with half a draw, forced tactical value, teacher agreement, or playing strength. The report
always states that conditioning. The owner must accept this approximation for the first model;
a meaningful draw rate requires a versioned target/serving decision before qualification.

Exclude unfinished/aborted/unknown-outcome games, corrupt states, states after king capture,
nonstandard rules, and records with ambiguous pre/post-turn perspective at the enrichment stage.
The bundle validator rejects missing/bad outcomes, inconsistent identities, non-finite features,
wrong engine/schema/order/digests, and material/perspective mismatches. The producer is responsible
for engine legality, canonical FEN, game-semantic exclusions and the truth of provenance. A hash
binds a claim to bytes; it cannot establish source rights or prove that training used those bytes.

All rows from a game, and every augmentation, transposition family or root expansion derived from
it, share one `group_id`. If roots connect several games, use their connected component. Each row
also carries `game_id` and optional `root_id`; the validator rejects aliases across groups.

## Data and provenance

A dataset directory contains exactly the following required inputs (extra files are ignored):

- `manifest.json`: `schema=playground-rows-v1`, semantic `version`, `kind` (`synthetic`,
  `public-licensed`, or `owner-controlled`), `target=decisive-game-outcome`,
  `perspective=side-to-move`, `feature_schema=kcp-13`, exact `columns`, `engine_version=0.9.2`,
  `rows_sha256`, `source_sha256`, `golden_sha256`, `license`, `license_evidence_sha256`.
- `rows.json`: array of objects with unique `id`, `group_id`, `game_id`, optional `root_id`,
  engine-canonical `fen`, `side` (`w`/`b`), nonnegative integer `ply`, `result` (0/0.5/1),
  and 13 engine-generated `features`. The seven material columns are checked by the existing
  test oracle; Python never derives the remaining features.
- `license.txt`: actual reviewed license/grant or owner-controlled-use evidence. Its byte digest
  is checked. A missing license or a software license alone is not permission for third-party data.

`source_sha256` binds the producer's immutable input/reproduction record. The data owner retains
that record for review; the public report contains none of its content. `golden_sha256` must match
the committed JVM-generated fixture, and `rows_sha256` binds every row and feature. A different
engine or schema requires an independently reviewed protocol/contract update; #17's S1/S2 cannot
be served or silently qualified as S0 merely because they share its prefix.

The public fixture in `tests/fixtures/benchmark` contains only authored synthetic positions from
the golden corpus (all `sample-*` probes are excluded) with artificial alternating binary labels.
Its data terms are explicit in `license.txt`. It tests mechanics and has **no model-quality
meaning**. The existing `sample/playsite-bots-v0` README does not provide an explicit third-party
redistribution grant; that sample is ineligible for this benchmark until separately reviewed
license evidence is supplied. This change neither republishes its rows nor changes that sample.

## Splits, repeats and holdout custody

Assignment is `int(SHA256("playground-v1:" + group_id), 16) % 10000`: values below 8000 are train,
8000–8999 validation, and 9000–9999 test. It is independent of row order and unrelated added groups.
Training/config selection uses train and validation only. The development command scores
**validation only**; it never emits test scores, test prevalence, test predictions, or test rows.
It reports split sizes and exact-position overlaps using canonical FEN placement, active side,
castling and en passant, ignoring clocks and dice. Producers must normalize unusable EP and other
state equivalences with the engine before export. This is an exact-position audit, not proof of
absence of near duplicates or collisions of the compressed 13-feature representation.

Development reports both all-validation and positions unseen in training. Shared openings may
appear across games; they remain visible in overlap counts. Final qualification uses an additional
owner-controlled sealed bundle, held **outside Git**, whose groups/game/root aliases cannot occur
anywhere in development. Any exact-position overlap with any development partition prevents
qualification; redraw/removal of final rows after seeing results is forbidden. All supplied final
rows are scored after the declared draw exclusion; they are not hashed into new splits.

Before training, the owner records the protocol Git commit/digest, development and final manifest
digests, group/exclusion policy, the reference inventory, and serving envelope in an access-controlled
append-only record. The trainer receives only the train/validation exports. The final custodian
retains the development test rows and the sealed final bundle. Before opening final labels, the
owner registers the single validation-selected candidate manifest digest in a sealed JSON record
and supplies its independently retained byte SHA-256 to the command.
`implementation_digest()` additionally binds the runner source, contract implementation and
`uv.lock`; final evaluation refuses a different implementation even if policy JSON is unchanged. This is the last lock, not an
opportunity to change earlier benchmark choices. Repeated final inspection/tuning invalidates the
holdout; log every invocation in the owner's immutable audit system.

The CLI verifies identity and disjointness. It cannot enforce filesystem access separation,
chronology, honest attestations, or prevent an owner making a new seal; those are custody/review
requirements. Never use a fresh CLI-computed seal digest as a substitute for the preregistered one.

## Scores and slices

Scores weight rows equally, while confidence intervals resample whole groups with replacement
(1000 draws, seed 13, percentile 2.5/97.5 bounds). Candidate/reference deltas use paired draws.
This preserves within-game dependence; it does not correct selection bias or make synthetic rows
independent real games. Fewer than two groups produces an explicit unsupported interval.

- Log-loss is mean `-y log(p) - (1-y) log(1-p)`, clipping only for logs at `1e-15`.
- Brier is mean `(p-y)^2`; prevalence is mean `y`.
- ECE uses ten equal-width bins, left inclusive/right exclusive except the last includes 1.
  It is the row-count-weighted absolute difference of mean prediction and prevalence per bin.
  Empty bins have count zero and null means. CIs include prevalence, log-loss, Brier and ECE;
  paired CIs include log-loss and Brier deltas.

The fixed slices cover phase (opening ply <= 10; otherwise endgame total material <= 20;
otherwise middlegame), side, material (behind < -1, balanced [-1,1], ahead > 1 in pawn units),
and four tactical feature flags (> 0 for king attack/danger and queen attack/danger).
All binary outcomes are `target:decisive`. `target:uncertain` is explicitly unsupported because
observed binary outcomes contain no teacher confidence. Do not classify uncertainty by the
candidate's own probabilities. Agreement/ranking is unsupported without teacher probabilities
or root-associated candidate order labels; inventing rank metrics from game outcomes is forbidden.
Every unsupported slice is reported. Missing critical slices block qualification; the predeclared
unsupported uncertain-target slice does not.

## Baselines and promotion gate

The no-information predictor is the decisive **training** prevalence (never validation or final
prevalence). The initial accepted deployable-reference list is empty: the synthetic serving fixture
and historical artifacts with incomplete lineage cannot be a promotion reference. The sealed
registry must name every accepted deployable reference by canonical manifest digest and nominate
one before candidate results; all registered references must be supplied with `--reference` and
are evaluated. When the list is nonempty, a deployable reference must be the promotion comparator.
V1 reference artifacts must have audited provenance bound to the same development bundle and train
groups as the candidate. A reference with a different training corpus needs an independently
reviewed leakage audit and protocol extension, not a silent exception.

The committed gate requires, on both the full final set and exact-unseen subset:

- At least 30 independent groups, >= 1% relative log-loss improvement, paired 95% log-loss-delta
  CI upper bound < 0, no Brier regression, and ECE regression <= 0.01.
- Each supported critical slice has at least 10 groups, log-loss regression <= 0.01 and Brier
  regression <= 0.01 against the predeclared comparator. These are point-estimate tolerances;
  slice intervals are diagnostic, not a simultaneous familywise confidence guarantee.
- Golden invariants and the sealed serving envelope pass. No final/development exact-position
  leakage, no missing provenance, no omitted accepted reference, no synthetic qualification.

These are prospective benchmark design constants, not production tuning values or measured
verdicts. Any future adjustment creates a new protocol before a new candidate/holdout evaluation.
A passing result says `eligible-for-owner-review`; it never deploys, merges, or claims arena strength.
Development always says `not-qualified`, even with perfect synthetic scores.

## Golden and serving probes

The authored portion of the JVM golden corpus is reused without editing its feature values:

| Probe | Expected invariant and scope |
| --- | --- |
| `start-w`, `start-b`, `start-w-6field` | identical predictions; within 0.1 of 0.5 (engineering sanity bound, not a theorem about first-move advantage) |
| `knight-up-w`, `start-w`, `knight-down-b` | strictly descending predicted probabilities (material sanity) |
| Every non-sample `*-twin` | same mover-canonical prediction, tolerance 1e-5; **not** `1-p` |
| `ep-e6-w`, `ep-none-w` | identical prediction after canonicalization |
| `rook-king-attack-w`, `rook-king-danger-w` | engine feature checks for one-face capture probability 91/216; a dice-free threat is not a certain win/loss |
| `rook-queen-attack-w`, `queen-en-prise-w` | correct attack/danger feature and twin invariants; no unproved probability ordering between different-material boards |
| Immediate legal king capture / forced terminal loss | legal capture returns mover win 1 without a leaf call; the captured side has terminal outcome 0, verified through engine/adapter evidence |

The [serving probe catalog](serving-probes-v1.json) fixes a legal capture witness and the
captured-side terminal-loss invariant. This does not assert that a nonterminal threatened king
has leaf probability zero. The existing `test_kcp13_contract.py` independently checks feature invariants. Candidate predictions
check canonical and material invariants. Piece-safety behavior and terminal/turn semantics require
owner evidence from the real evaluator, not a Python approximation or treating the leaf network
as a move-search oracle. The evidence must include the concrete legal dice/turn fixtures and raw
observations for capture, forced loss and matched piece-safety alternatives, in the private audit.

A serving evidence JSON has schema `playground-serving-evidence-v1`, the exact
`candidate_manifest_sha256`, `probe_suite_sha256` (byte digest of the committed serving catalog),
`raw_evidence_sha256`, `concurrency_workload_sha256`, boolean `checks` for `jvm_golden_parity`
(max error <= 1e-5), `torch_onnx_parity` (<= 1e-6), `immediate_king_capture`, `forced_loss`,
`piece_safety`, `probability_bounds`, `concurrency`, and finite nonnegative `measurements`
(`latency_p95_ms`, `rss_mb`). Compare to positive limits in the preregistered seal. Limits and raw
service details remain private; reports emit only an evidence digest and failed check names.
The suite contains the normative acceptance criteria for every boolean, including parity bounds,
raw probability bounds and concurrent-versus-serial replay. The concurrency workload manifest
freezes load levels, request mix/order and sample accounting privately before results; its digest
must match between seal and serving evidence. The booleans attest reviewed external evidence;
the CLI does not run the JVM or attest its truth.

## Commands and model card

From a clean checkout:

```bash
mise run setup
uv run python -m dicechess_training.benchmark --data tests/fixtures/benchmark
mise run check
```

For a development candidate, add `--candidate <candidate-directory>`. It contains `model.onnx`
and the ADR 0001 `manifest.json` plus required provenance: `training_data_sha256` (canonical
**development manifest** digest), `training_groups_sha256` (canonical sorted train group list),
`config_sha256`, `code_sha256` (SHA-256 source snapshot), `benchmark_sha256`, `engine_version`,
integer `seed`, `perspective=side-to-move`. `train_identity(manifest, rows)` produces the two
training identity fields. The config and source snapshot remain available to the reviewer outside
Git. Calibration must be inside ONNX; a manifest-only temperature other than 1 is rejected.

### Producing a candidate package

`dicechess_training.candidate` builds such a directory instead of assembling it by hand:

```bash
uv run python -m dicechess_training.candidate \
  --data <dataset-directory> --output <candidate-directory> \
  --seed 11 --model-id <identifier> --report <private-summary.json>
```

It admits the dataset through `load_dataset` before training, splits with the benchmark's own
`assignments`, trains on the train partition only, selects the epoch budget on an inner split
carved out of that partition (groups whose split value is at least 7000), folds the training
statistics into the exported graph as constants, and fills the provenance block with digests
computed from the artifacts — `training_data_sha256` and `training_groups_sha256` come from
`train_identity`, so `check_training_identity` passes by construction rather than by hand.

Before anything is written it runs the checks the benchmark will run (`validate_manifest`,
`verify_model_digest`, `validate_onnx_contract`, `check_training_identity`) plus Torch-versus-
onnxruntime parity on the golden matrix and held-out rows at 1e-6; on any failure the output
directory stays empty. It refuses to overwrite an existing package. `model-card.md` is written
next to the artifacts with the fields that can be read from them; the private rows of the card
stay the owner's to complete outside Git. Like the benchmark entry point, stdout carries only
the JSON summary — digests and counts, never a dataset path or row content.

The seal has schema `playground-seal-v1`, `implementation_sha256`, `benchmark_sha256`, `development_dataset_sha256`,
`final_dataset_sha256`, `candidate_manifest_sha256`, `accepted_references` (digest list),
`promotion_reference` (an accepted digest, or `no-information` only with no accepted models),
`concurrency_workload_sha256`, and `serving_limits` for the measurements above. `digest()` canonicalizes JSON with sorted keys,
compact separators and UTF-8; file/evidence/model byte digests use SHA-256 of exact bytes. Retain
the canonical dataset manifest digest as the final dataset identity; it binds the row-byte digest.

```bash
uv run python -m dicechess_training.benchmark \
  --mode final --data <sealed-final-directory> --development-data <development-directory> \
  --candidate <candidate-directory> --seal <preregistered-seal.json> \
  --seal-sha256 <independently-retained-sha256> \
  --serving-evidence <reviewed-serving-evidence.json> \
  --output <private-report.json>
# Add --reference <reference-directory> for every accepted reference.
```

Exit 0 means a development report completed or final eligibility passed; final refusal exits 1;
invalid input/runtime/IO exits 2 with a sanitized JSON error, never an input path or traceback.
`--output` exclusively creates a new file; existing files and symlinks are refused. Input and output
paths are selected by the invoking owner of this local CLI; this is not a remote service or an
agent tool with a privileged filesystem boundary. Keep any external automation within its own
file-access authorization policy.
Keep all real candidate reports, weights, final rows, seals and experiment verdicts outside public
Git. Public reports may include only synthetic mechanics; sanitized identities are schema/version/
digest, never model names, host names, storage locations, credentials, group IDs or FENs.

[report-schema-v1.json](report-schema-v1.json) defines the machine-readable report. The report
serves as the quantitative attachment to the [private model-card template](model-card-template.md). The private owner-reviewed model card must also
record intended use, the decisive-only limitation, data terms/exclusions/custody, training source
and config digests, accepted comparator and seal identity, golden/serving evidence, failed slices,
and the owner decision. Numerical thresholds must not be changed in the model card.
