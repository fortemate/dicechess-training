"""Train == serve contracts: the exact feature layouts, tensor names, manifest rules and
probability perspectives that a served model must satisfy, mirrored fail-closed on the Python
side. Feature *values* are never reimplemented here; the engine is their single source of truth
and the golden corpora under ``tests/fixtures/`` carry its answers."""

from __future__ import annotations

from . import kcp13, kcp_mobility27, kcp_mobility_pawns31, prerank

# `prerank` is deliberately absent below: this mapping answers "which value-model contract
# reads an enriched shard with this feature schema and exports under its tensor names", and a move
# pre-ranker is neither a value model nor a consumer of those shards.
SCHEMA_CONTRACTS = {
    kcp13.SCHEMA_ID: kcp13,
    kcp_mobility27.SCHEMA_ID: kcp_mobility27,
    kcp_mobility_pawns31.SCHEMA_ID: kcp_mobility_pawns31,
}

__all__ = [
    "SCHEMA_CONTRACTS",
    "kcp13",
    "kcp_mobility27",
    "kcp_mobility_pawns31",
    "prerank",
]
