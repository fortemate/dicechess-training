"""Convert a `dicechess-analytics` training export into schema-v0 Parquet shards.

The analytics export speaks its own dialect and this module is the single place
that translates it:

* `result` arrives in the stored White-POV encoding (`1` / `0` / `-1` for a White
  win / draw / loss). Schema v0 stores the MOVER's perspective as 0.0 / 0.5 / 1.0,
  so the score is rescaled and then flipped for Black's turns.
* `dice_sorted` is upper-case on White's turns and lower-case on Black's — the
  case redundantly repeats `side`, so it is normalized to upper case and the
  multiset is what survives.
* `normalized_fen` carries placement, active color, castling, and en passant
  (no move counters); only the placement field feeds the encoder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import validate_dice, write_shard

EXPORT_COLUMNS = ("game_id", "turn_number", "fen", "dice", "side", "result")
WHITE_POV_RESULTS = (-1, 0, 1)


def convert_export(df: pd.DataFrame) -> pd.DataFrame:
    """Translate an analytics export frame into schema-v0 columns."""
    missing = [c for c in EXPORT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"export is missing columns: {missing}")
    bad_result = set(df["result"].dropna().unique()) - set(WHITE_POV_RESULTS)
    if bad_result:
        raise ValueError(f"result must be White-POV {WHITE_POV_RESULTS}, got {sorted(bad_result)}")

    white_score = (df["result"].astype("float32") + 1.0) / 2.0
    is_black = df["side"] == "b"
    return pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "ply": df["turn_number"].astype("int32"),
            "fen": df["fen"].astype(str),
            "dice": df["dice"].map(validate_dice),
            "side": df["side"].astype(str),
            "result": white_score.where(~is_black, 1.0 - white_score).astype("float32"),
        }
    )


def convert_export_file(csv_path: str, out_dir: str, shard_size: int = 25_000) -> list[str]:
    """Convert an export CSV (optionally gzipped) into schema-v0 shards.

    Shards are split on game boundaries so a single game never straddles two
    files — game-level splitting stays possible shard by shard.
    """
    df = convert_export(pd.read_csv(csv_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = Path(csv_path).name.split(".")[0]

    paths: list[str] = []
    start = 0
    while start < len(df):
        end = min(start + shard_size, len(df))
        if end < len(df):  # extend to the end of the game straddling the boundary
            game = df["game_id"].iloc[end - 1]
            while end < len(df) and df["game_id"].iloc[end] == game:
                end += 1
        path = str(out / f"{prefix}-{len(paths):03d}.parquet")
        write_shard(df.iloc[start:end].reset_index(drop=True), path)
        paths.append(path)
        start = end
    return paths
