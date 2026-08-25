import pandas as pd
import pytest

from dicechess_training import ingest, schema

EXPORT = pd.DataFrame(
    {
        "game_id": ["g1", "g1", "g2", "g2"],
        "turn_number": [1, 2, 1, 2],
        "fen": [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -",
            "4k3/8/8/8/8/8/8/4K3 w - -",
            "4k3/8/8/8/8/8/8/4K3 b - -",
        ],
        "dice": ["BBK", "bbn", "QRR", "qrr"],
        "side": ["w", "b", "w", "b"],
        "result": [1, 1, 0, 0],  # White-POV: g1 White win, g2 draw
    }
)


def test_result_is_translated_to_mover_perspective():
    df = ingest.convert_export(EXPORT)
    # White win: 1.0 for White's turn, 0.0 for Black's turn.
    assert df.loc[0, "result"] == pytest.approx(1.0)
    assert df.loc[1, "result"] == pytest.approx(0.0)
    # Draw: 0.5 for both sides.
    assert df.loc[2, "result"] == pytest.approx(0.5)
    assert df.loc[3, "result"] == pytest.approx(0.5)


def test_dice_case_is_normalized():
    df = ingest.convert_export(EXPORT)
    assert list(df["dice"]) == ["BBK", "BBN", "QRR", "QRR"]


def test_converted_frame_satisfies_schema():
    schema.validate_frame(ingest.convert_export(EXPORT))


def test_rejects_unknown_result_encoding():
    bad = EXPORT.copy()
    bad.loc[0, "result"] = 2
    with pytest.raises(ValueError, match="White-POV"):
        ingest.convert_export(bad)


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="turn_number"):
        ingest.convert_export(EXPORT.drop(columns=["turn_number"]))


def test_shards_never_split_a_game(tmp_path):
    csv = tmp_path / "export.csv"
    EXPORT.to_csv(csv, index=False)
    paths = ingest.convert_export_file(str(csv), str(tmp_path / "shards"), shard_size=1)
    assert len(paths) == 2  # one per game, despite shard_size=1
    for path in paths:
        assert schema.read_shard(path)["game_id"].nunique() == 1


def test_rejects_non_positive_shard_size(tmp_path):
    csv = tmp_path / "export.csv"
    EXPORT.to_csv(csv, index=False)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="shard_size"):
            ingest.convert_export_file(str(csv), str(tmp_path / "shards"), shard_size=bad)


def test_interleaved_games_are_never_split(tmp_path):
    """An export that interleaves games must still yield whole games per shard."""
    interleaved = EXPORT.iloc[[0, 2, 1, 3]].reset_index(drop=True)
    assert list(interleaved["game_id"]) == ["g1", "g2", "g1", "g2"]
    csv = tmp_path / "interleaved.csv"
    interleaved.to_csv(csv, index=False)

    paths = ingest.convert_export_file(str(csv), str(tmp_path / "shards"), shard_size=1)

    seen = set()
    for path in paths:
        shard = schema.read_shard(path)
        assert shard["game_id"].nunique() == 1
        game = shard["game_id"].iloc[0]
        assert game not in seen, f"game {game} was split across shards"
        seen.add(game)
        assert list(shard["ply"]) == sorted(shard["ply"])
    assert seen == {"g1", "g2"}
