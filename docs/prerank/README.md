# The move pre-ranker's corpus

How training data for the learned move pre-ranker (#9) is made, and why it is made in two
repositories rather than one.

## What a pre-ranker is for

The engine's search cannot afford to evaluate every legal turn. A three-die turn has a few hundred
of them, and the evaluation worth having — the champion's, which asks over all 216 rolls how often
each king can actually be taken — costs milliseconds per candidate. So the search ranks everything
cheaply, keeps the best `candidateLimit` (48 in production), and spends the expensive evaluation
only on that shortlist.

The pre-ranker is the cheap ranker. It is trained to put the turns the expensive evaluation would
have liked at the top of the list, so the shortlist contains them. It is a **student** of that
evaluation, which is the fact that decides everything below.

## Why two repositories

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

## Running it

```bash
# 1. choose the roots, and see what labelling them will cost before paying it
uv run python -m dicechess_training.prerank roots <shards-dir> <work-dir> --limit 5000

# 2. label them (in dicechess-hunter)
mise run prerank:corpus -- <work-dir>/roots.tsv configs/profiles/baseline-v1.json <corpus-dir>

# 3. write the manifest and admit the result
uv run python -m dicechess_training.prerank pack <corpus-dir> \
    --license-file <terms.txt> --license "Fortemate owner-controlled; not for redistribution"
```

Step 1 writes `roots.tsv` — tab-separated `game_id, fen, dice, side` — and
`roots-provenance.json`, which records the shards it read with their content digests. The roots
file is the producer's _input_ record, which is what `source_sha256` binds. Rebuilding the corpus
needs the rest of the run as well: the generator's commit, the engine it resolved and the profile
it ran with, which `generation.json` names. Between the two, a corpus can be traced back to the
shards it came from and forward to the evaluation that labelled it —
`roots-provenance.json` is the audit trail for the first half and is not needed to regenerate
anything once `roots.tsv` exists.

Step 3 writes `manifest.json` beside `groups.json`, `generation.json` and `license.txt`. It cannot
write a manifest the loader would reject, because it runs the loader before returning.

## Three rules worth knowing before you run it

**One root, once.** A group's id is derived from its position and its roll, so the same pair twice
is the same list twice, which `load_groups` refuses. This is ordinary rather than exotic: in the
committed sample, 11% of rows repeat a root and 704 roots occur in more than one game. The
exporter deduplicates, charging each root to its earliest `(game_id, ply)` occurrence — which also
decides the split it lands in, so it has to be a rule rather than an accident. It also removes the
leak a game-level split cannot see, because a root that exists once cannot land on both sides.

**A bigger sample contains the smaller one.** Roots are ordered by a hash of their own identity,
and `--limit` takes a prefix. Raising the limit adds roots without disturbing the ones already
chosen, so a larger corpus reuses the core-hours already spent instead of invalidating them.

**Compute is not the constraint; the file is.** Measured at engine 0.12.0 on 400 roots of the
sample: 120.7 candidates per root read, 0.60 ms per candidate on eight threads, and **235 bytes of
JSON per candidate**.

| roots                     | candidates | JSON   | CPU            |
| ------------------------- | ---------- | ------ | -------------- |
| 5,000                     | 0.60M      | 142 MB | 0.8 core-hours |
| 43,692 (the whole sample) | 5.28M      | 1.2 GB | 7 core-hours   |

A gigabyte of JSON is a gigabyte the loader holds as Python objects. Size the first corpus around
5,000 roots; wanting a much larger one is a reason to revisit
`playground-prerank-groups-v1`'s storage format, not the machine it runs on.

## What the corpus looks like

From the end-to-end run of 400 sampled roots (engine 0.12.0, teacher `hunter-baseline-v1`):

- 25 roots were a forced pass — no legal turn — and produced no group;
- 375 groups, 48,298 candidates, and **173,422 turn paths collapsed as transpositions**: 78% of
  everything the rules allow reaches a position another path already reached;
- candidates per group: min 1, median 43, p95 520, max 1,642;
- **180 of 375 groups are larger than the shortlist of 48.** In the other 195 a pre-ranker cannot
  be wrong, because every candidate survives — so a rank metric has to name the subset it was
  measured on;
- the median group has **17 distinct target values**, so the teacher discriminates. This is the
  question that killed the idea of a king-capture-probability leaf teacher, whose median group was
  entirely tied.

## Related

- `docs/decisions/0001-playground-train-serve-contract.md` — the train/serve contract this follows.
- `dicechess_training.prerank.groups` — the `playground-prerank-groups-v1` contract itself.
- `dicechess_training.contracts.prerank` — what the trained artifact must be.
- `dicechess-hunter`, `docs/prerank-corpus.md` — the generator, and what it costs.
