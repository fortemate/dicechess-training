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

600 root rows from the first corpus, of which **576** have a legal turn; every legal turn path
applied; four costs timed on the same states:

- **material** — a material pass, one `Evaluator.evaluateMaterial` per candidate: what the engine
  ships and what this replaces. (`ExpectimaxSearch.materialBatch` itself is `private[search]`, so
  this is the same arithmetic through the public surface rather than that method's own number.)
- **features** — `RichFeatures.extract` per candidate, which material does not pay.
- **model** — one ONNX Runtime call on the batch.
- **added** — the whole extract-and-score pass timed as _one_ measurement, minus material. The two
  components are reported for diagnosis but never summed: their minima can fall in different
  repetitions, so a sum of them can be smaller than any pass that actually happened.

Best of 15 repeats per root. Every repeat is identical work, so anything above the floor is the
machine doing something else; the minimum is the machine's answer and the mean would be the
laptop's.

### Environment

```bash
docker run --rm --cpus=1 --memory=512m -v "$PWD:/bench:ro" -w /tmp eclipse-temurin:25-jre \
  java -Xmx384m -cp "/bench/lib/classes:/bench/lib/*" \
  dicechess.training.latency.PreRankLatency /bench/model.onnx /bench/roots.tsv 15
```

One vCPU, 512 MB cgroup, **384 MB heap** — deliberately below the cgroup so ONNX Runtime's native
allocations have room. The project's own `javaOptions` use the same heap, so `sbt run` does not
quietly measure a two-gigabyte machine the report does not describe.

### One ORT thread

| candidates | roots | material |  features |    model |     added | added per candidate |
| ---------: | ----: | -------: | --------: | -------: | --------: | ------------------: |
|        ≤ 8 |    76 |    0.2µs |     4.7µs |    6.1µs |    11.0µs |             2.50 µs |
|       ≤ 16 |    26 |    0.3µs |    13.0µs |    8.6µs |    21.8µs |             1.79 µs |
|       ≤ 32 |    48 |    0.6µs |    27.4µs |   12.9µs |    39.1µs |             1.65 µs |
|       ≤ 64 |    47 |    1.1µs |    53.7µs |   21.8µs |    75.2µs |             1.53 µs |
|      ≤ 128 |    60 |    2.1µs |   108.1µs |   39.6µs |   146.3µs |             1.52 µs |
|      ≤ 256 |    70 |    4.3µs |   234.4µs |   74.3µs |   304.5µs |             1.59 µs |
|      ≤ 512 |    54 |    8.0µs |   456.8µs |  134.4µs |   581.8µs |             1.53 µs |
|     ≤ 1024 |    72 |   15.0µs |   983.8µs |  256.7µs |  1227.4µs |             1.68 µs |
|     ≤ 4096 |   103 |   37.6µs |  2531.2µs |  634.4µs |  3112.3µs |             1.56 µs |
|     > 4096 |    20 |  146.9µs | 10312.0µs | 2403.2µs | 12438.5µs |             1.83 µs |
|  **total** |   576 |          |           |          |           |                     |

Median root, 180 candidates: **0.344 ms**. Widest root, 16,965 candidates: **29.8 ms**.

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
sbt "run <model.onnx> <roots.tsv> 15"
```

The repeat count is the third argument and defaults to 20; the table above was taken at 15, so
pass it explicitly to reproduce those numbers. For the published environment rather than a
developer laptop, use the container command in **Environment** above — `sbt run` forks a JVM with
the same 384 MB heap but none of the CPU or memory limits.

Pinned to engine 0.12.0 rather than parameterised: the artifact declares
`engineCompatibility >=0.12.0`, and measuring the seam against an engine that does not have the
seam would measure something else. Writes `prerank-latency.json` with every root's sample.
