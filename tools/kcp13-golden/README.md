# kcp13-golden

Generates the `kcp-13` golden corpus in `tests/fixtures/kcp13/` by replaying `probes.tsv` through
the released `dicechess-engine` artifact's `KcpFeatures.extract` — the same code the private
evaluation service runs at serving time. See
[ADR 0001](../../docs/decisions/0001-playground-train-serve-contract.md).

Requires a JDK (21+) and sbt; it is not part of `mise run check`, the fixture is committed.

```bash
mise run golden:kcp13                 # engine 0.9.2 -> tests/fixtures/kcp13/golden-engine-0.9.2.json
ENGINE_VERSION=0.8.3 mise run golden:kcp13 -- /tmp/golden-0.8.3.json   # replay against another release
```

Extraction latency per probe is printed to stdout and deliberately kept out of the fixture so
that two engine versions produce byte-comparable files.
