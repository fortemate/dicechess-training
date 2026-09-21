# Pre-ranker, protocol v1, first result

Five seeds of `playground-prerank-v1` on the first corpus — 5,000 roots of the 100k development
corpus, 4,789 groups, 752,504 candidates, teacher `hunter-baseline-v1`, engine 0.12.0.

**The learned ranker beats the ordering the engine ships at the tightest shortlist, decisively.
At the wider ones it leads by the same direction and the corpus cannot yet say it is real.** Reported on the validation partition; the test partition is
reserved and was not looked at.

## Recall of a best candidate

"Did the shortlist keep a turn the teacher rated best?" — which is the question the seam asks,
because the search then rescores everything it kept and takes the maximum.

| shortlist | groups | learned (5 seeds) | `material_diff` | random | difference | paired 95% CI | wins / losses | exact _p_, per seed |
| --------- | -----: | ----------------: | --------------: | -----: | ---------: | ------------- | ------------: | ------------------: |
| 8         |    399 |   0.6426 ± 0.0065 |          0.5238 | 0.2516 |   +11.9 pp | [+6.6, +17.2] | 83–85 / 34–40 |    0.00001 – 0.0001 |
| 16        |    364 |   0.7418 ± 0.0049 |          0.6868 | 0.3352 |    +5.5 pp | [−0.2, +10.8] | 58–60 / 37–41 |       0.032 – 0.107 |
| 48        |    264 |   0.8682 ± 0.0044 |          0.8182 | 0.4780 |    +5.0 pp | [+0.5, +10.1] | 28–29 / 14–16 |       0.032 – 0.096 |

`rank1` — the student's own first choice holding the teacher's maximum — is 0.2598 ± 0.0039
against 0.2045 for `material_diff` and 0.0152 for random.

Each width is measured only on groups larger than it, so the three rows do not share a
denominator: a group of 20 candidates can be ranked wrong at a shortlist of 8 and cannot be at 48.

**What is and is not established.** At a shortlist of 8 the result is not in doubt: every seed
gives _p_ ≤ 0.0001. At 16 and 48 it is not established. Across the five seeds the same test
ranges from 0.032 to 0.107 — it lands on both sides of 0.05 depending on which seed's weights are
being measured, which is the definition of a result a sample this size cannot resolve.

The direction is the same at all three widths, the effect grows as the shortlist tightens, and
that is the shape the mechanism predicts. But "+5 pp at the production width" is a number this
corpus cannot separate from zero, and it must not be quoted without that sentence.

**Why two tests.** The protocol preregistered a bootstrap, and the bootstrap is primary. It is
also, at these sample sizes, sensitive to how many resamples were drawn: the interval at a
shortlist of 16 excluded zero at 2,000 resamples and includes it at the preregistered 1,000. That
is not a reason to prefer 2,000 — it is a reason to report something that does not depend on the
draw. The discordant counts are the same comparison with nothing left to chance, because only the
groups where the two orderings disagree carry information about which is better, and the exact
two-sided binomial on them is McNemar's test without an approximation.

An earlier version of this page quoted single _p_ values of 0.054 and 0.049, computed from the
counts averaged across seeds and then tested. That is the wrong way round: averaging counts and
then testing understates how much the answer moves between seeds. The ranges above are what the
seeds actually produced.

## What the numbers mean, and what they do not

**The baseline is the pre-ranker being replaced, not an arbitrary column.** Ordering by
`material_diff` _is_ `ExpectimaxSearch.materialBatch`, the default `preRank` of the engine's ONNX
search. So for that family of bots the comparison is like for like, and the width where the gain
is clearest — 8 — is a width that search actually uses.

**It is not the baseline for the champion.** `dicechess-hunter` ranks its first phase with
`HunterEval.cheapScore`, which is much stronger than material alone, and the corpus carries only
the expensive target so that ordering cannot be measured here. Nothing in this table says a
learned ranker would improve _that_ bot. Measuring it means a second label in the corpus and a
regeneration.

**A wide shortlist forgives a bad ordering.** A random order already keeps a best candidate 48% of
the time at a shortlist of 48, because the evaluated groups have a median of 173 candidates and
48 of them is a generous cut. That is why the random row is here: without it, 0.87 looks like
skill when roughly half of it is the width. It is also why the advantage shrinks as the shortlist
widens — there is less left to win.

**The uncertainty is in the corpus, not in the training.** The recall figures barely move
between seeds — ±0.5 pp — while the paired standard error is ±2.5 to 2.7 pp, and the _p_ value
swings across 0.05 because only 44 of 264 groups discriminate between the two orderings at a
shortlist of 48 at all. More seeds cannot fix that; more validation groups can.

**Validation, not test.** The protocol reserves the test partition, and the source corpus reserves
a temporal holdout beyond that; neither was touched. This corpus is development data by its own
manifest, so these numbers can guide development and cannot qualify anything.

## Training behaviour

Four seconds per seed on a laptop. The loss plateaus by the fourth epoch or so and the best
validation epoch lands between 3 and 7, which is worth remembering when the corpus grows: the
capacity chosen for serving cost is not the thing currently limiting the result.

## What has to happen before this is worth serving

1. **A larger corpus**, which is now the binding constraint on the measurement rather than on the
   model. 5,000 roots gave 264 groups at the production width and only 44 that discriminate.
2. **Latency at the seam.** The model scores every legal turn before the search consults its
   deadline — up to 2,420 at one root. A 32×32 MLP was chosen for that reason and has never been
   measured there.
3. **A fixed-time arena.** Recall of a teacher's choice is not strength. The only thing that can
   say this makes a bot win more games is a game.
4. **The probe-pair gate.** Authored position pairs differing only in whether a piece hangs, to
   check the ranker orders them the obvious way — carried over from `dicechess-ev#1`.
5. **`cheapScore` in the corpus**, if the champion rather than the ONNX family is the target.
