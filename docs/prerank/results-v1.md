# Pre-ranker, protocol v1, first result

Five seeds of `playground-prerank-v1` on the first corpus — 5,000 roots of the 100k development
corpus, 4,789 groups, 752,504 candidates, teacher `hunter-baseline-v1`, engine 0.12.0.

**The learned ranker beats the ordering the engine ships, at every shortlist width, and by most
where the shortlist is tightest.** Reported on the validation partition; the test partition is
reserved and was not looked at.

## Recall of a best candidate

"Did the shortlist keep a turn the teacher rated best?" — which is the question the seam asks,
because the search then rescores everything it kept and takes the maximum.

| shortlist | groups | learned (5 seeds) | `material_diff` | random | difference | paired 95% CI |
| --------- | -----: | ----------------: | --------------: | -----: | ---------: | ------------- |
| 8         |    399 |   0.6426 ± 0.0065 |          0.5238 | 0.2516 |   +11.9 pp | [+6.6, +17.3] |
| 16        |    364 |   0.7418 ± 0.0049 |          0.6868 | 0.3352 |    +5.5 pp | [+0.1, +10.9] |
| 48        |    264 |   0.8682 ± 0.0044 |          0.8182 | 0.4780 |    +5.0 pp | [+0.5, +9.9]  |

`rank1` — the student's own first choice holding the teacher's maximum — is 0.2598 ± 0.0039
against 0.2045 for `material_diff` and 0.0152 for random.

Each width is measured only on groups larger than it, so the three rows do not share a
denominator: a group of 20 candidates can be ranked wrong at a shortlist of 8 and cannot be at 48.

## What the numbers mean, and what they do not

**The baseline is the pre-ranker being replaced, not an arbitrary column.** Ordering by
`material_diff` _is_ `ExpectimaxSearch.materialBatch`, the default `preRank` of the engine's ONNX
search. So for that family of bots the comparison is like for like, and the widths where the gain
is largest — 8 and 16 — are the widths that search actually uses.

**It is not the baseline for the champion.** `dicechess-hunter` ranks its first phase with
`HunterEval.cheapScore`, which is much stronger than material alone, and the corpus carries only
the expensive target so that ordering cannot be measured here. Nothing in this table says a
learned ranker would improve _that_ bot. Measuring it means a second label in the corpus and a
regeneration.

**A wide shortlist forgives a bad ordering.** A random order already keeps a best candidate 48% of
the time at a shortlist of 48, because the evaluated groups have a median of 173 candidates and
48 of them is a generous cut. That is why the random row is here: without it, 0.87 looks like
skill when roughly half of it is the width.

**The uncertainty is in the corpus, not in the training.** Seed spread is ±0.5 pp; the paired
interval is ±5 pp. Both are honest, and they say the same thing: more seeds would not sharpen
this, more validation groups would. At a shortlist of 16 the interval's lower bound is +0.1 pp —
the effect is there, but only just, and it should not be quoted as "about 5 points" without it.

**Validation, not test.** The protocol reserves the test partition, and the source corpus reserves
a temporal holdout beyond that; neither was touched. This corpus is development data by its own
manifest, so these numbers can guide development and cannot qualify anything.

## Training behaviour

Four seconds per seed on a laptop. The loss plateaus by the fourth epoch or so and the best
validation epoch lands between 3 and 7, which is worth remembering when the corpus grows: the
capacity chosen for serving cost is not the thing currently limiting the result.

## What has to happen before this is worth serving

1. **Latency at the seam.** The model scores every legal turn before the search consults its
   deadline — up to 2,420 at one root. A 32×32 MLP was chosen for that reason and has never been
   measured there.
2. **A fixed-time arena.** Recall of a teacher's choice is not strength. The only thing that can
   say this makes a bot win more games is a game.
3. **The probe-pair gate.** Authored position pairs differing only in whether a piece hangs, to
   check the ranker orders them the obvious way — carried over from `dicechess-ev#1`.
4. **`cheapScore` in the corpus**, if the champion rather than the ONNX family is the target.
