# Pre-ranker, protocol v1, first result

Five seeds of `playground-prerank-v1` on the first corpus — 5,000 roots of the 100k development
corpus, 4,789 groups, 752,504 candidates, teacher `hunter-baseline-v1`, engine 0.12.0.

**The learned ranker beats the ordering production actually serves, at every shortlist width,
by 6:1 to 11:1 in discordant pairs.** Reported on the validation partition; the test partition is
reserved and was not looked at.

## Recall of a best candidate

"Did the shortlist keep a turn the teacher rated best?" — which is the question the seam asks,
because the search then rescores everything it kept and takes the maximum.

| shortlist | groups | learned (5 seeds) | `oracle-3` | `material_diff` | random | vs `oracle-3` | vs `material_diff` |
| --------- | -----: | ----------------: | ---------: | --------------: | -----: | ------------: | -----------------: |
| **8**     |    399 |   0.6426 ± 0.0065 |     0.5539 |          0.5238 | 0.2732 |    41 W / 6 L |        83 W / 36 L |
| 16        |    364 |   0.7418 ± 0.0049 |     0.6346 |          0.6868 | 0.3791 |    43 W / 4 L |        58 W / 38 L |
| **24**    |    335 |   0.7904 ± 0.0061 |     0.6955 |          0.7522 | 0.4299 |    38 W / 6 L |        45 W / 32 L |
| 48        |    264 |   0.8682 ± 0.0044 |     0.7917 |          0.8182 | 0.5189 |    22 W / 2 L |        28 W / 15 L |

Wins and losses are discordant groups, averaged over the seeds. The worst seed's exact two-sided
test against `oracle-3` is _p_ ≤ 0.0002 at every width; against `material_diff` it is ≤ 0.0001 at
a shortlist of 8 and 0.107, 0.314 and 0.096 at 16, 24 and 48.

The bold widths are deployed: `dexus-atlas-1` runs `ORACLE_CANDIDATE_LIMIT=8`, and
`dicechess-bot-gcp-onnx` and `-4` run 24. 16 is not deployed anywhere; it is kept because it is
where the engine's own scaladoc measured its `candidateLimit` step.

The full metric table, including the rank-1 and rank-2 hit rates and the list metric, is below.

Each width is measured only on groups larger than it, so the four rows do not share a
denominator: a group of 20 candidates can be ranked wrong at a shortlist of 8 and cannot be at 48.

**What is and is not established.** Against `oracle-3` — the ordering `dexus-atlas-1` actually
serves — the result is not in doubt at any width: 6:1 to 11:1 in discordant pairs and every seed's
exact test at or below 0.0002.

Against `material_diff` only the shortlist of 8 is established (_p_ ≤ 0.0001 on every seed). At
16, 24 and 48 the worst seed gives 0.107, 0.314 and 0.096 — the direction is consistent and the
corpus cannot resolve it. That comparison still matters, because material is what the two older
ONNX bots rank with, and at their width of 24 it is the weaker of the two claims.

**`oracle-3` is worse than plain material at three of the four widths.** 0.6346 against 0.6868 at
16, 0.6955 against 0.7522 at 24, 0.7917 against 0.8182 at 48; only at 8 is it ahead, 0.5539 to
0.5238. Each deployed bot happens to be on the better side of that — `dexus-atlas-1` sets
`PRE_RANK_WITH_MODEL=true` and runs at 8, the other two leave it off and run at 24 — but the
margin at 8 is three points and nobody chose it on evidence. **This is a flag, not a verdict**:
`oracle-3` is a win-probability model and the teacher here is hunter's evaluation, so the two
optimise different things, and a search whose leaf _is_ `oracle-3` has an argument for a shortlist
ranked by `oracle-3`. Worth its own measurement; see the open questions below.

## Every metric, at every width — the exported artifact

**These are seed 11's numbers, measured through the exported `model.onnx`,** not the five-seed
aggregate of the table above. That is deliberate: this is the artifact, and an artifact is what
gets served, gated and benchmarked. It is also why the figures differ slightly from the aggregate
— 0.6491 against 0.6426 ± 0.0065 at a shortlist of 8, which is seed 11 sitting one standard
deviation above the mean of five.

Each width is measured on the groups where a shortlist of that size can be wrong — `size > k` —
so the rows do not share a denominator. Rank-1 and rank-2 are recall at widths 1 and 2 on their
own groups.

|      width | groups | recall: learned | `oracle-3` | material | random | NDCG: learned | `oracle-3` | material |
| ---------: | -----: | --------------: | ---------: | -------: | -----: | ------------: | ---------: | -------: |
| 1 (rank-1) |    473 |      **0.3362** |     0.2664 |   0.2833 | 0.0867 |        0.3882 |     0.3106 |   0.3253 |
| 2 (rank-2) |    463 |      **0.4471** |     0.3650 |   0.3499 | 0.1166 |        0.4124 |     0.3346 |   0.3364 |
|          8 |    399 |      **0.6491** |     0.5539 |   0.5238 | 0.2732 |        0.4659 |     0.3822 |   0.3727 |
|         16 |    364 |      **0.7445** |     0.6346 |   0.6868 | 0.3791 |        0.4949 |     0.4040 |   0.4110 |
|         24 |    335 |      **0.7910** |     0.6955 |   0.7522 | 0.4299 |        0.5124 |     0.4221 |   0.4250 |
|         48 |    264 |      **0.8682** |     0.7917 |   0.8182 | 0.5189 |        0.5188 |     0.4240 |   0.4436 |

> An earlier version of this page quoted `rank1` as 0.2598. That figure was measured on the groups
> larger than 48 — the hardest subset — because the training report reused one denominator for
> every metric. On its own groups the rank-1 hit rate is **0.3362**. The number was not wrong, it
> was answering a narrower question than its name suggests.

### The list metric resolves what recall cannot

The same artifact — seed 11 — against `material_diff`, paired on the same groups:

| width | recall difference | 95% CI        | exact _p_ | NDCG difference | 95% CI           |
| ----: | ----------------: | ------------- | --------: | --------------: | ---------------- |
|     1 |           +5.3 pp | [+1.3, +9.3]  |     0.019 |          +0.063 | [+0.024, +0.100] |
|     2 |           +9.7 pp | [+5.2, +14.0] |     0.000 |          +0.076 | [+0.041, +0.111] |
|     8 |          +12.5 pp | [+7.3, +17.8] |     0.000 |          +0.093 | [+0.061, +0.125] |
|    16 |           +5.8 pp | [+0.3, +10.7] |     0.040 |          +0.084 | [+0.049, +0.118] |
|    24 |           +3.9 pp | [−1.2, +8.7]  |     0.171 |          +0.087 | [+0.056, +0.120] |
|    48 |           +5.3 pp | [+0.8, +10.2] |     0.049 |          +0.075 | [+0.041, +0.109] |

**At a shortlist of 24 recall cannot tell the two orderings apart and NDCG can.** Recall's interval
crosses zero and its exact test gives 0.171; NDCG's interval is nowhere near zero at any width.

That is not a contradiction, it is the difference between the two questions. Recall is binary and
largely saturated — at 48 both orderings keep a best candidate four times in five, so only the
discordant minority carries any information, and there are 44 such groups. NDCG uses where _every_
candidate landed, so the same 335 groups carry far more of it.

So the honest summary is two sentences rather than one. **By the list metric the learned ranker is
better than the ordering the engine ships at every width tested.** Whether it more often keeps a
best candidate _inside the cut_ is established up to a shortlist of 16 and not resolved at 24,
where this corpus runs out of discordant groups.

### Calibration

Deliberately absent, and the definition of done's "where meaningful" is the reason. An ordering is
invariant to every monotone transform of the scores, so there is no quantity a calibration curve
could be drawn against: the model is never asked how good a candidate is, only which of two is
better. `contracts.prerank` makes the same point structurally by refusing a `calibration` block on
this role — a model that needed one would be a value model wearing the wrong name.

What is usually wanted from calibration here is "how good is the whole list, not just the top
pick", and that is what NDCG answers.

## What the numbers mean, and what they do not

**The baselines are orderings that are actually served.** `oracle-3` is the leaf value model
`dexus-atlas-1` pre-ranks with — fetched from the lab bucket and verified against the
`MODEL_SHA256` that service declares, `5e3cd9a1…`. It consumes the same nine columns this corpus
carries, so it is measured here rather than approximated. `material_diff` is
`ExpectimaxSearch.materialBatch`, the engine's default, which is what the two older ONNX bots use
because they do not set `PRE_RANK_WITH_MODEL`.

**Still not the baseline for the champion.** `dicechess-hunter` ranks its first phase with
`HunterEval.cheapScore`, which the corpus does not carry, so nothing here says a learned ranker
would improve that bot. Measuring it means a second label and a regeneration.

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

## The artifact

Seed 11 is exported as `prerank-dev-5k-seed11`: `model.onnx` of 12,851 bytes and a manifest 1.1.0
declaring `modelRole: move-prerank`, `featureSchema: rich-9-v1`, `featureCount: 9` and
`modelSha256 bdb9300d…`. No `calibration` block — an ordering has nothing to calibrate, and the
contract refuses one on this role.

**The numbers above are the artifact's, not the checkpoint's.** Training ran in float64 and the
contract's tensors are FLOAT, so the export narrows the arithmetic, and a comparison between two
nearly equal scores can flip when it does. Re-measured through the exported graph over all 752,504
candidates: scores agree with the checkpoint to 4.4e-06, and **not one group changes its outcome
at any of the three widths**. On the engine's own golden probes the graph and the model agree to
9.5e-07.

The bytes are a function of the model rather than of the machine: exporting the same checkpoint
twice gives the same digest, because the absolute source paths `torch.onnx.export(dynamo=True)`
writes onto every node are stripped — the defect that left every package built before #60
unrecoverable.

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
