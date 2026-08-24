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
