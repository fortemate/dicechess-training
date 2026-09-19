# Engine manifest fixtures

Copied byte-for-byte from `dicechess-engine` (`jvm/src/test/resources/`, engine #78 / PR #254) so
this repository's Python contract is tested against the manifests the JVM reader actually accepts,
rather than against manifests written by the same code that validates them.

`tests/test_kcp13_contract.py` asserts both sides agree on `manifestVersion`, `modelRole`,
`perspective` and the tensor names, and records the one field they deliberately differ on:
`evaluationProfile`. The engine does not read it — it wires a model into its own search — so its
1.1.0 fixtures omit it. The deployed evaluation service selects its serving profile with it, and
this contract mirrors the service, so a manifest without it is refused here. Packages built by
`kcp13.build_manifest` always carry it and therefore satisfy both readers.

Regenerate by copying again; do not edit in place.
