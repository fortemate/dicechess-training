"""The shard reader rejects incompatible semantics before exposing features or targets."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dicechess_training.contracts import SCHEMA_CONTRACTS, kcp13
from dicechess_training.schema import COLUMNS, read_enriched_shard


@pytest.fixture(params=list(SCHEMA_CONTRACTS))
def shard(request):
    sid = request.param
    fields = list(COLUMNS.items()) + [
        (name, pa.float32()) for name in SCHEMA_CONTRACTS[sid].COLUMN_NAMES
    ]
    values = ["synthetic-game", 0, "4k3/8/8/8/8/8/8/4K3 w - -", "PPP", "w", 1.0]
    values += [np.float32(0)] * len(SCHEMA_CONTRACTS[sid].COLUMN_NAMES)
    metadata = {
        b"feature_schema": sid.encode(),
        b"engine_version": b"0.9.3",
        b"ruleset": kcp13.RULESET_VERSION.encode(),
        b"perspective": kcp13.PERSPECTIVE.encode(),
    }
    table = pa.Table.from_arrays(
        [pa.array([v], type=t) for v, (_, t) in zip(values, fields, strict=True)],
        schema=pa.schema(fields, metadata=metadata),
    )
    return sid, table


@pytest.mark.parametrize("field", [b"ruleset", b"perspective"])
@pytest.mark.parametrize("value", [None, b"", b"wrong", b"white"])
def test_invalid_semantics_rejected(shard, tmp_path, field, value):
    sid, table = shard
    metadata = dict(table.schema.metadata)
    if value is None:
        del metadata[field]
    else:
        metadata[field] = value
    path = tmp_path / "incompatible.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match=f"shard {field.decode()}"):
        read_enriched_shard(path, sid, "0.9.3")


def test_compatible_semantics_load(shard, tmp_path):
    sid, table = shard
    path = tmp_path / "compatible.parquet"
    pq.write_table(table, path)
    assert len(read_enriched_shard(path, sid, "0.9.3")) == 1
