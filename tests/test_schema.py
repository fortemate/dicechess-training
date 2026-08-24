import pandas as pd
import pyarrow.parquet as pq
import pytest

from dicechess_training import data, schema


def test_shard_round_trip(tmp_path):
    df = data.generate_toy_games(5, seed=1)
    path = str(tmp_path / "shard.parquet")
    schema.write_shard(df, path)
    loaded = schema.read_shard(path)
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(schema.COLUMNS)


def test_unknown_schema_version_rejected(tmp_path):
    df = data.generate_toy_games(2, seed=2)
    path = str(tmp_path / "shard.parquet")
    schema.write_shard(df, path)
    table = pq.read_table(path).replace_schema_metadata({schema.SCHEMA_METADATA_KEY: b"v999"})
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="v999"):
        schema.read_shard(path)


def test_dice_validation():
    assert schema.validate_dice("nqk") == "NQK"
    with pytest.raises(ValueError):
        schema.validate_dice("NQ")
    with pytest.raises(ValueError):
        schema.validate_dice("NQX")


def test_frame_validation_rejects_bad_result():
    df = data.generate_toy_games(2, seed=3)
    df.loc[0, "result"] = 0.7
    with pytest.raises(ValueError, match="result"):
        schema.validate_frame(df)


def test_frame_validation_rejects_missing_column():
    df = data.generate_toy_games(2, seed=4).drop(columns=["dice"])
    with pytest.raises(ValueError, match="dice"):
        schema.validate_frame(pd.DataFrame(df))
