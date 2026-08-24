"""Training data schema v0.

Shards are Parquet files carrying provenance columns only; feature encodings are
computed at load time (see `features`), so encodings can evolve without
regenerating data. Mirrors the `dicechess-analytics` export vocabulary:
`game_id, fen, dice, side, result`.

`result` is stored from the MOVER's perspective: 1.0 = the side to move went on
to win the game, 0.0 = lost, 0.5 = draw. Positions within one game share an
outcome, which is why splits must be game-level (see `data.split_by_game`).

The `dice` column is assumed to be three piece letters from `PIECE_TYPES`
(order-insensitive multiset, e.g. "NQK"). This assumption must be verified
against the real analytics export before the public sample lands (issue #4).
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "v0"
SCHEMA_METADATA_KEY = b"dicechess_training_schema"

PIECE_TYPES = "PNBRQK"
DICE_COUNT = 3
SIDES = ("w", "b")
RESULTS = (0.0, 0.5, 1.0)

COLUMNS: dict[str, pa.DataType] = {
    "game_id": pa.string(),
    "ply": pa.int32(),
    "fen": pa.string(),
    "dice": pa.string(),
    "side": pa.string(),
    "result": pa.float32(),
}


def validate_dice(dice: str) -> str:
    """Return the dice string normalized to upper case, or raise ValueError."""
    normalized = dice.upper()
    if len(normalized) != DICE_COUNT or any(c not in PIECE_TYPES for c in normalized):
        raise ValueError(f"dice must be {DICE_COUNT} letters from {PIECE_TYPES!r}, got {dice!r}")
    return normalized


def validate_frame(df: pd.DataFrame) -> None:
    """Validate a shard DataFrame against schema v0. Raises ValueError on violations."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for column in COLUMNS:
        if df[column].isna().any():
            raise ValueError(f"column {column!r} contains nulls")
    if (df["fen"].astype(str).str.strip() == "").any():
        raise ValueError("column 'fen' contains empty strings")
    if (df["ply"] < 0).any():
        raise ValueError("column 'ply' contains negative values")
    bad_side = set(df["side"].unique()) - set(SIDES)
    if bad_side:
        raise ValueError(f"side must be one of {SIDES}, got {sorted(bad_side)}")
    bad_result = set(df["result"].unique()) - set(RESULTS)
    if bad_result:
        raise ValueError(
            f"result must be one of {RESULTS} (mover perspective), got {sorted(bad_result)}"
        )
    for dice in df["dice"].unique():
        validate_dice(dice)


def write_shard(df: pd.DataFrame, path: str) -> None:
    """Validate and write one Parquet shard, stamping the schema version."""
    validate_frame(df)
    table = pa.Table.from_pandas(
        df[list(COLUMNS)].astype({"ply": "int32", "result": "float32"}),
        schema=pa.schema(list(COLUMNS.items())),
        preserve_index=False,
    )
    table = table.replace_schema_metadata({SCHEMA_METADATA_KEY: SCHEMA_VERSION.encode()})
    pq.write_table(table, path)


def read_shard(path: str) -> pd.DataFrame:
    """Read one Parquet shard, rejecting unknown schema versions loudly."""
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    version = metadata.get(SCHEMA_METADATA_KEY, b"").decode()
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: shard schema version {version!r} != supported {SCHEMA_VERSION!r}"
        )
    for name, expected in COLUMNS.items():
        if table.schema.get_field_index(name) < 0:
            raise ValueError(f"{path}: missing column {name!r}")
        actual = table.schema.field(name).type
        if actual != expected:
            raise ValueError(f"{path}: column {name!r} has type {actual}, expected {expected}")
    df = table.to_pandas()
    validate_frame(df)
    return df
