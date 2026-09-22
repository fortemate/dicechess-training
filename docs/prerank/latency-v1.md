# Pre-ranker latency at the seam

What a learned pre-ranker costs on every turn, measured on real candidate sets with
`tools/prerank-latency`.

**It is affordable.** On a median root the pass adds **0.35 ms**, 0.017% of a 2,000 ms turn budget.
On the widest root in the sample — 16,965 legal turn paths — it adds **30 ms**, 1.5%.

## Why this number decides something

The pre-rank pass sees **every** legal turn and is paid in full _before_ the search consults its
deadline, because its output is what the anytime fallback plays. The cost is not amortised over a
search that might finish early: it is subtracted from the turn budget up front, every turn, at the
full branching factor. That is why the schema was chosen for cost and the model kept at 32×32.

## What was measured

600 roots from the first corpus, every legal turn path applied, three costs timed on the same
states: a material pass (what the engine ships), `RichFeatures.extract` per candidate (which
material does not pay), and one ONNX Runtime call on the batch. The added cost is
`features + inference − material`. Best of 15 repeats per root — every repeat is identical work,
so anything above the floor is the machine doing something else.

### One vCPU, 512 MB container, `eclipse-temurin:25-jre`, one ORT thread

| candidates | roots | material | features |   model |    added | added per candidate |
| ---------: | ----: | -------: | -------: | ------: | -------: | ------------------: |
|        ≤ 8 |    76 |    0.2µs |    5.1µs |   6.0µs |   10.9µs |             2.47 µs |
|       ≤ 32 |    48 |    0.6µs |   28.0µs |  12.8µs |   40.3µs |             1.70 µs |
|      ≤ 128 |    60 |    2.1µs |  113.7µs |  39.0µs |  150.6µs |             1.56 µs |
|      ≤ 512 |    54 |    8.0µs |  477.2µs | 134.3µs |  603.5µs |             1.59 µs |
|     ≤ 4096 |   103 |   36.7µs | 2459.0µs | 625.7µs | 3048.0µs |             1.53 µs |

Median root, 180 candidates: **0.345 ms**. Widest root, 16,965 candidates: **29.7 ms**.

The unpinned run is within noise of the pinned one and sometimes slower — at this size the model
does not repay thread coordination, so pinning ORT to one thread costs nothing and removes a way
for the pass to compete with the search for cores.

On the host workstation (Apple silicon, unconstrained) the same measurement gives 1.2–1.4 µs per
candidate and 23 ms on the widest root: the same picture, about 15% faster.

## Three things worth carrying forward

**Feature extraction is the cost, not the model.** `RichFeatures.extract` is roughly 80% of what
is added; the inference is 20%. Making the network bigger is nearly free at this scale, and any
future attempt to make the pass cheaper should start with the extractor. It also means the cost of
this seam is mostly _not_ a function of the model that sits in it.

**For a bot that already pre-ranks with a model, the increment is nothing.** `dexus-atlas-1` sets
`PRE_RANK_WITH_MODEL=true`, so it already pays the extraction and an inference of its own. For
that bot the cost of swapping this model in is the difference between two inferences. The table is
the pessimistic case — the cost of replacing the _material_ pre-ranker, which is what the two
older ONNX bots run.

**The candidate counts are raw turn paths, not deduplicated decisions.** A root whose corpus group
holds 2,420 candidates hands this pass up to 17,000 states, because 78% of turn paths are
transpositions and the corpus collapses them while the seam may not. Whether the engine
deduplicates before pre-ranking is not visible from here, so this measures the pessimistic reading
of that too.

## What is not measured

**x86.** This ran on Apple silicon, in a container, and the bots run on Cloud Run. The lab's x86
host was offline when this was measured, so the architecture the production bots actually use is
still untested. The margin is large enough — 1.5% of budget at the worst root — that a 5×
slower machine would still fit, but that is an argument, not a measurement.

**A real turn.** This times the pass in isolation. It does not observe a bot playing, where the
pass competes with the search for the same cores and the same cache.

## Reproducing

```bash
cd tools/prerank-latency
sbt "run <model.onnx> <roots.tsv> [repeats]"
```

Pinned to engine 0.12.0 rather than parameterised: the artifact declares
`engineCompatibility >=0.12.0`, and measuring the seam against an engine that does not have the
seam would measure something else. Writes `prerank-latency.json` with every root's sample.
