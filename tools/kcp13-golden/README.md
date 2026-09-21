# kcp13-golden

Replays the public probe definitions through the released engine extractors.
The module contains the golden-vector generator, the Parquet enrichment producer,
and the extraction-latency benchmark. See [ADR 0001](../../docs/decisions/0001-playground-train-serve-contract.md).

Use **JDK 21** and sbt for enrichment. The Hadoop dependency does not support
newer JDKs (reproduced on JDK 26) (`Subject.getSubject`); select JDK 21 with `JAVA_HOME`.

Which schemas a build offers follows the engine it resolves. `kcp-13` and
`rich-9-v1` replay against every engine this module can resolve; the mobility
schemas (`kcp-mobility-27-v1`, `kcp-mobility-pawns-31-v1`), the enrichment
producer and the extraction benchmark need engine **0.9.3 or later**, which is
the release that first published `KcpMobilityFeatures`. Selecting an older
engine compiles the `scala-legacy` tier instead of `scala-mobility`, so
`-Dengine.version=0.9.1` replays the two cheap schemas rather than failing to
compile over a class that engine never had. Asking such a build for a mobility
schema fails at startup, naming the schemas it does offer.

From the repository root:

```bash
mise run golden:kcp13 -- ../../out/golden-engine-0.9.3.json
mise run check:enrichment
```

The golden task selects `Kcp13Golden` explicitly and defaults to engine 0.9.3.
An explicit golden output path is resolved from `tools/kcp13-golden`; use an
absolute path or `../../out/golden-engine-0.9.3.json` for repository-root output.
Without an output argument it regenerates the versioned fixture.
The enrichment smoke creates temporary synthetic input, runs the JVM producer for
S0/S1/S2, and checks the output metadata and all feature vectors with the Python
contracts. It also verifies rejection and cleanup of a side/FEN mismatch. CI runs
this smoke independently of the Python checks.

To enrich an audited schema-v0 source, from `tools/kcp13-golden`:

```bash
sbt -batch "runMain dicechess.training.golden.EnrichShardsApp <input-dir> <output-dir> <schema-id>"
```

Supported schema IDs are `kcp-13`, `kcp-mobility-27-v1`, and
`kcp-mobility-pawns-31-v1`. Output declares both the ruleset and side-to-move
perspective; readers reject legacy shards missing those fields. Regenerate such
shards from verified source data rather than restamping unknown features.

Extraction latency is kept out of golden feature vectors. For measurement artifact
validation and publication rules, see the [ablation protocol](../../docs/ablation/README.md).
