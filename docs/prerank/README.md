# The move pre-ranker

How the learned move pre-ranker (#9) is built, trained, measured, exported and served — and what
the artifact is allowed to claim when it gets there.

## What a pre-ranker is for

The engine's search cannot afford to evaluate every legal turn. A three-die turn has a few hundred
of them, and the evaluation worth having — the champion's, which asks over all 216 rolls how often
each king can actually be taken — costs milliseconds per candidate. So the search ranks everything
cheaply, keeps the best `candidateLimit` (48 in production), and spends the expensive evaluation
only on that shortlist.

The pre-ranker is the cheap ranker. It is trained to put the turns the expensive evaluation would
have liked at the top of the list, so the shortlist contains them. It is a **student** of that
evaluation, which is the fact that decides everything below.

## The whole path

| step                | command                           | where                |
| ------------------- | --------------------------------- | -------------------- |
| choose the roots    | `prerank roots`                   | `dicechess-training` |
| enumerate and label | `mise run prerank:corpus`         | `dicechess-hunter`   |
| write the manifest  | `prerank pack`                    | `dicechess-training` |
| train and measure   | `prerank train`                   | `dicechess-training` |
| export the artifact | `prerank export`                  | `dicechess-training` |
| gate the artifact   | `prerank probe-pairs`             | `dicechess-training` |
| decide strength     | `dicechess-hunter/referee/run.sh` | `dicechess-hunter`   |

```bash
# 1. choose the roots, and see what labelling them will cost before paying it
uv run python -m dicechess_training.prerank roots <shards-dir> <work-dir> --limit 5000

# 2. label them (in dicechess-hunter)
mise run prerank:corpus -- <work-dir>/roots.tsv configs/profiles/baseline-v1.json <corpus-dir>

# 3. write the manifest and admit the result
uv run python -m dicechess_training.prerank pack <corpus-dir> \
    --license-file <terms.txt> --license "Fortemate owner-controlled; not for redistribution"

# 4. run the protocol's five seeds
uv run python -m dicechess_training.prerank train <corpus-dir> <runs-dir>

# 5. export one run's weights as an artifact the engine's seam can open
uv run python -m dicechess_training.prerank export <runs-dir>/weights-seed-11.pt <artifact-dir> \
    --model-id <name> --provenance protocol=playground-prerank-v1

# 6. ask whether it puts a hanging queen below its safe twin
uv run python -m dicechess_training.prerank probe-pairs <artifact-dir>/model.onnx
```

**Two commands return 2 rather than 0 on a bad answer, which is not the same as failing.** `train`
does it when a run does not clear the ordering the engine already ships, and `probe-pairs` when the
ranker does not separate a blunder from the move that avoids it. Both are results. A refusal — a
corpus that will not admit, a file that cannot be read — is 1, and its message says what went
wrong and never where, because a corpus's location is as private as its contents.

## Why the middle step is in another repository

| step                | where                                    | why there                                                             |
| ------------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| choose the roots    | `dicechess-training` (`prerank.roots`)   | sampling, deduplication and splits are this repository's rules        |
| enumerate and label | `dicechess-hunter` (`PrerankGroupsMain`) | the teacher is that bot's own evaluation, and its weights never leave |
| write the manifest  | `dicechess-training` (`prerank.pack`)    | a manifest states digests this repository defines and checks          |

The middle step is in the private bot repository because the label is
`SearchScoring.scorePath(root, path, HunterEval.fullScore)` — not a re-implementation of the
champion's evaluation but the literal expression its search uses. Training against anything else
would train a student of a bot that does not exist.

The last step is here because a manifest has to state the canonical-JSON digest of a golden corpus
committed here, under a convention defined here. Restating that convention in Scala would put a
digest format in two places: the two would agree until a float was formatted differently, and the
symptom would be every corpus refused with no indication why. So the generator writes what only it
can know, and this side turns that into a manifest — and then admits the result through
`load_groups`, the same loader every consumer uses.

## The corpus

Step 1 writes `roots.tsv` — tab-separated `game_id, fen, dice, side` — and
`roots-provenance.json`, which records the shards it read with their content digests. The roots
file is the producer's _input_ record, which is what `source_sha256` binds. Rebuilding the corpus
needs the rest of the run as well: the generator's commit, the engine it resolved and the profile
it ran with, which `generation.json` names.

Step 3 writes `manifest.json` beside `groups.json`, `generation.json` and `license.txt`. It cannot
write a manifest the loader would reject, because it runs the loader before returning.

### Three rules worth knowing before you run it

**One root, once.** A group's id is derived from its position and its roll, so the same pair twice
is the same list twice, which `load_groups` refuses. This is ordinary rather than exotic: in the
committed sample, 11% of rows repeat a root and 704 roots occur in more than one game. The
exporter deduplicates, charging each root to its earliest `(game_id, ply)` occurrence — which also
decides the split it lands in, so it has to be a rule rather than an accident. It also removes the
leak a game-level split cannot see, because a root that exists once cannot land on both sides.

**A bigger sample contains the smaller one.** Roots are ordered by a hash of their own identity,
and `--limit` takes a prefix. Raising the limit adds roots without disturbing the ones already
chosen, so a larger corpus reuses the core-hours already spent instead of invalidating them.

**Compute is not the constraint; the file is.** Measured on the first real run — 5,000 roots of
the 100k development corpus at engine 0.12.0, eight threads: 150.5 candidates per root read, 0.55
ms per candidate, and **237 bytes of JSON per candidate**.

| roots                           | candidates | JSON   | CPU            |
| ------------------------------- | ---------- | ------ | -------------- |
| 5,000                           | 0.75M      | 178 MB | 0.9 core-hours |
| 1,231,068 (every distinct root) | 185M       | 43 GB  | 226 core-hours |

A gigabyte of JSON is a gigabyte the loader holds as Python objects, so the file format binds long
before the machine does. `prerank roots` projects this before the run — but the rate depends on
the corpus: the committed public sample averages 120.7 candidates per root against the development
corpus's 150.5, which is why the first projection came in 26% under. Re-measure when the source
changes rather than trusting a number drawn from a different one.

### What the first corpus looked like

5,000 roots of the 100k development corpus, engine 0.12.0, teacher `hunter-baseline-v1`, retained
on the run host as `20260921-prerank-corpus-dev-5k`:

- 211 roots were a forced pass — no legal turn — and produced no group (4.2%);
- 4,789 groups, 752,504 candidates, and **2,738,029 turn paths collapsed as transpositions**:
  78.4% of everything the rules allow reaches a position another path already reached;
- candidates per group: min 1, median 61, p95 643, max 2,420;
- **2,643 of 4,789 groups are larger than the shortlist of 48** (55%). In the other 45% a
  pre-ranker cannot be wrong, because every candidate survives — so a rank metric has to name the
  subset it was measured on;
- the median group has **26 distinct target values** and only 4.2% are tied throughout, so the
  teacher discriminates. This is the question that killed the idea of a king-capture-probability
  leaf teacher, whose median group was entirely tied.

A corpus built from that development source is **development data**: its own manifest calls it
"ineligible for final qualification", and the reserved temporal holdout stays untouched. Train,
compare and measure on it; qualify elsewhere.

## Training

`prerank train <corpus-dir> <runs-dir>` runs the protocol against an admitted corpus. There is
almost nothing to choose, on purpose — `docs/prerank/protocol-v1.json` was written before the first
run and the command is that document executed:

|           |                                                                                 |
| --------- | ------------------------------------------------------------------------------- |
| objective | softmax cross-entropy over the whole list (ListNet), against rank-derived gains |
| gains     | `1 / rank²` on the teacher's ordering, normalised per group                     |
| network   | `9 → 32 → 32 → 1`, standardisation held inside the model as buffers             |
| optimiser | Adam, `lr = 1e-3`, 64 groups per batch                                          |
| stopping  | up to 40 epochs, patience 5 on validation recall, best-epoch weights restored   |
| shortlist | `k = 48`, the production `candidateLimit`                                       |
| seeds     | 11, 23, 47, 89, 131 — five runs, reported as a range                            |

The gains are the one decision worth restating here, because it is where a listwise loss usually
goes wrong. They come from the _rank_ the teacher gave a candidate, never from its score: targets
are in the teacher's private units, their spread varies by orders of magnitude between groups, and
a softmax over raw values would be a per-group temperature nobody chose. `1 / rank²` also needs no
special case for a king capture — that candidate carries `TerminalWinScore` (2³¹−1), which ranks
first and is never arithmetic — and it turns a wholly-tied group into a uniform target, which asks
the model to be flat there rather than to invent an order.

Each seed writes `report-seed-<n>.json` and `weights-seed-<n>.pt`. The checkpoint is a bare
`state_dict`: `export` rebuilds the architecture from the shapes alone, so a run's output can be
reopened with nothing beside it.

**A run that does not beat `material_diff` is a negative result, and the command says so in its
exit status.** That column is the engine's own shipped pre-ranker, `ExpectimaxSearch.materialBatch`,
and it costs nothing because it is already one of the nine features. The ablation of #17 produced
arms worse than a constant and only an admissibility floor caught it; this is the same floor, put
where a pipeline can see it. The report is written either way — a negative result that is not
written down is a negative result somebody repeats.

Four seconds per seed on a laptop, against 752,504 candidates. The best validation epoch lands
between 3 and 7, which is worth remembering when the corpus grows: the capacity chosen for serving
cost is not what currently limits the result.

## Evaluation

The same command reports it, and `prerank.metrics` computes it for an ordering that was never
trained here — a deployed ONNX model, or a single feature column — so a baseline that is actually
served can be put on the same footing.

**Each width has its own denominator, inside one split.** Two subsettings compose here, and which
is which matters, because one shortlist yields a different group count depending on the population
it is applied to. First the split: metrics are reported on the **validation** groups, 489 of the
first corpus's 4,789. Then the width: a ranker can only be wrong in a group with more candidates
than the shortlist keeps, so each width is measured on the validation groups with `size > k`.

|   k | validation groups measured |
| --: | -------------------------: |
|   1 |                        473 |
|   2 |                        463 |
|   8 |                        399 |
|  16 |                        364 |
|  24 |                        335 |
|  48 |                        264 |

So the 264 here and the 2,643 in the corpus section are the same rule applied to different
populations — 2,643 of all 4,789 groups exceed a shortlist of 48, and 264 of the 489 validation
groups do. Every reported number carries its own group count for exactly this reason: collapsing
the widths onto one denominator would make rank-1 look harder than it is and recall-at-48 easier.

**Ties are settled by value, never by index.** A group can hold several candidates the teacher
scored identically and any of them is a right answer; scoring against one chosen index would mark
a right answer wrong whenever the teacher was indifferent.

| what                                          | why it is reported                                                                                                                             |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| recall at 1, 2, 8, 16, 24, 48                 | 1 and 2 are the rank-1 and rank-2 hit rates; 8 and 24 are what the deployed bots run; 48 is production's `candidateLimit`                      |
| NDCG at the same widths                       | recall asks whether a best candidate survived the cut; NDCG asks how good the cut is, and moves when the second and third move                 |
| `material_diff`                               | the ordering the engine ships, and the floor                                                                                                   |
| `random`                                      | the lower bound — a random order keeps a best candidate half the time at 48, which says how much of any figure is the shortlist being generous |
| paired bootstrap, 1,000 resamples at seed 13  | both orderings are measured on the same groups, so the interval belongs on their difference                                                    |
| exact two-sided binomial on discordant groups | McNemar without the chi-squared approximation; a comparison is decided entirely by the groups the two orderings disagree about                 |

NDCG uses the same gains the loss is trained against, so the objective and the report agree about
what a good list looks like instead of quietly optimising one thing and measuring another. It also
resolves what recall cannot: on the first corpus, at a shortlist of 24 recall's interval crosses
zero and NDCG's is nowhere near it.

**Calibration is not reported, because for an ordering it is not meaningful.** An ordering is
invariant to every monotone transform of the scores, so there is no quantity a calibration curve
could be drawn against — the model is never asked how good a candidate is, only which of two is
better. `contracts.prerank` makes the same point structurally by refusing a `calibration` block on
this role: a model that needed one would be a value model wearing the wrong name.

The first result is in `results-v1.md`, with every width, both baselines and the intervals.

## Export

`prerank export` writes `model.onnx` and `manifest.json`, validates the manifest against the graph
beside it, and prints the largest disagreement between the checkpoint and the exported graph on the
engine's own golden probes. Three things happen there that are not "save the model":

**The graph is float32 and the training was not.** The contract's tensors are FLOAT because that is
what the serving runtime binds, so the export narrows the arithmetic and then _re-measures_: a
ranking is decided by comparisons, and a comparison between two nearly equal scores can flip when
the arithmetic narrows. The numbers that belong in a report are the artifact's, not the
checkpoint's — on the first corpus they agree to 4.4e-06 and not one group changed its outcome at
any of the widths re-measured, but that is a measurement rather than an assumption.

**The bytes are a function of the model, not of the machine.** `torch.onnx.export(dynamo=True)`
writes the absolute source path and line number of the traced `forward` onto every node, which
made `modelSha256` depend on where the checkout happened to live and left packages built before #60
permanently unrecoverable. That metadata is stripped, so exporting the same checkpoint twice gives
the same digest.

**The batch axis stays dynamic.** The seam feeds every legal turn through the model in chunks of
`PreRankModel.DefaultChunkSize` (256 rows), so a graph pinned to a row count would refuse the only
workload it has.

The manifest is version 1.1.0 and declares `modelRole: move-prerank`, `featureSchema: rich-9-v1`,
`featureCount: 9`, `perspective: side-to-move` and `modelSha256`. It carries no `calibration`
block, for the reason above.

Then gate it. `prerank probe-pairs` scores authored position pairs that differ only in whether a
queen is left en prise and asks for `score(safe) > score(blunder)`. Read `probe-pairs-v1.md`
before quoting a pass rate from it — the reason is under **Deployment limits** below, and it is a
fact about the schema rather than about any model.

## Deployment limits

**The seam needs engine 0.12.1 or newer, whatever the manifest's range says.** `PreRankModel` does
not exist at v0.12.0 — it appears in `OnnxExpectimaxSearch.scala` at v0.12.1. The manifest's
`engineCompatibility` is about the _feature layout_: a 0.12.0 host can score these bytes, it simply
has nowhere to put them.

**The slot is exclusive.** `OnnxSearchOptions.preRankModel` and `preRankWithModel` configure the
same seam, and the engine refuses a host that sets both — deliberately, because picking a winner
would hide the mistake from the only person who knows which was meant. `preRankWithModel` reuses
the leaf model; a dedicated package is the alternative, not a companion.

**The pass is paid in full before the deadline is consulted**, because its output is what the
anytime fallback plays. In a one-CPU container at one ORT thread it costs about 1.5 µs per
candidate: 0.344 ms at the median root and 29.8 ms at the widest root measured, which carried
16,965 candidates. Feature extraction is ~80% of that and inference ~20%, so the cost of this seam
is mostly _not_ a function of the model sitting in it. The counts are raw turn paths — the corpus
collapses transpositions and the seam may not — so that is the pessimistic reading. `latency-v1.md`
has the table.

**x86 is unmeasured.** The benchmark ran on Apple silicon and the production bots run on Cloud Run.
The margin is wide enough that a 5× slower machine would still fit, but that is an argument, not a
measurement.

**No host can currently be told to use a dedicated pre-ranker.** `dicechess-bot-gcp-onnx` pins
engine 0.12.0 and its only switch is the `PRE_RANK_WITH_MODEL` boolean. Wiring is
`dicechess-bot-gcp-onnx#56`; the arena is blocked on it.

**The schema has no feature for a piece being en prise.** `rich-9` is exactly `kcp-13` without its
four capture-probability columns, which is what makes it affordable at this width — `kcp-13` at a
thousand candidates is a thousand 216-outcome searches and is not viable at this seam at all. So
no rich-9 ranker can pass the probe-pair gate for the right reason, and a pass rate from it is a
statement about `mobility_diff`.

**Strength is not established.** Recall of a teacher's choice is not strength: the pre-rank pass is
paid out of the same turn budget as the search, so a ranker that orders better and costs more can
lose. `arena-protocol-v1.json` preregisters the game that decides it. It has not been run.

**The data terms follow the artifact.** The corpus is owner-controlled and not approved for public
redistribution; a model trained on it inherits that, and a package built from the development
corpus is development-only — ineligible for qualification until the reserved holdout is used.

## Artifact provenance

Almost every step binds the one before it by digest. The one that does not is the last, and it is
named here rather than left to be discovered.

| link                    | what binds it                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| shards → roots          | `roots-provenance.json` records every shard read with its content digest                           |
| roots → corpus          | `generation.json` states `roots_sha256`; the manifest carries it as `source_sha256`                |
| corpus → its own bytes  | `groups_sha256`, and `license_evidence_sha256` for the terms beside it                             |
| corpus → feature layout | `golden_sha256`, the committed `rich9/golden-engine-<v>.json`                                      |
| corpus → run            | the training report copies `source_sha256`, `groups_sha256`, the teacher id and the engine version |
| artifact → its bytes    | `modelSha256` in the exported manifest binds it to the graph beside it                             |

**Run → artifact is a naming, not a binding.** Every row above is a digest; the step from the run
to the package it produced is four strings in the manifest's `provenance` —
`corpus: 20260921-prerank-corpus-dev-5k`, `seed: 11`, `teacher: hunter-baseline-v1`,
`protocol: playground-prerank-v1`. A label is not a digest, so an artifact on its own says which
corpus it _claims_ and not which corpus it _had_. Two mitigations, in order of preference:

1. pass the digest at export time — `provenance` is free-form, so
   `--provenance groups_sha256=<the corpus manifest's value>` closes the gap for new packages;
2. keep `report-seed-<n>.json` beside the package. The report carries both corpus digests, so a
   package plus its report is fully traceable even when the manifest only names things.

`prerank-dev-5k-seed11` predates the first of those and relies on the second.

**A published run re-runs to the same bytes.** `prerank train` over the first corpus with
`--seeds 11` writes a checkpoint byte-identical to the one `prerank-dev-5k-seed11` was exported
from — same digest, best epoch 7, validation recall 0.8712, on the host that produced it. Read
that as same-host reproduction: the seed pins the run, not the hardware, and a machine with
different floating-point behaviour may land on different last bits while reaching the same
conclusion. It is still the point of fixing the seeds in the protocol rather than passing them on
the command line — a seed chosen after seeing a result is not a replication of anything.

## Related

- `docs/decisions/0001-playground-train-serve-contract.md` — the train/serve contract this follows.
- `dicechess_training.prerank.groups` — the `playground-prerank-groups-v1` contract itself.
- `dicechess_training.contracts.prerank` — what the trained artifact must be.
- `docs/prerank/protocol-v1.json` and `docs/prerank/results-v1.md` — the preregistered protocol
  and the first result.
- `dicechess-hunter`, `docs/prerank-corpus.md` — the generator, and what it costs.
- `docs/prerank/latency-v1.md` — what the pass costs at the seam.
- `docs/prerank/probe-pairs-v1.md` — the hanging-queen gate, and what it says about the schema.
- `docs/prerank/arena-protocol-v1.json` — how strength gets decided, and what it is blocked on.
