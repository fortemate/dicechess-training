import pytest

from dicechess_training import data


def test_split_rejects_degenerate_inputs():
    df = data.generate_toy_games(1, seed=9)
    with pytest.raises(ValueError, match="at least 2 games"):
        data.split_by_game(df, val_fraction=0.2, seed=9)
    two = data.generate_toy_games(2, seed=9)
    for bad_fraction in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="val_fraction"):
            data.split_by_game(two, val_fraction=bad_fraction, seed=9)


def test_split_always_keeps_a_training_game():
    df = data.generate_toy_games(2, seed=10)
    train_df, val_df = data.split_by_game(df, val_fraction=0.9, seed=10)
    assert len(train_df) > 0
    assert len(val_df) > 0


def test_split_has_no_game_leakage():
    df = data.generate_toy_games(50, seed=5)
    train_df, val_df = data.split_by_game(df, val_fraction=0.2, seed=5)
    assert len(train_df) + len(val_df) == len(df)
    assert set(train_df["game_id"]).isdisjoint(set(val_df["game_id"]))
    assert len(val_df) > 0


def test_positions_within_game_share_outcome():
    df = data.generate_toy_games(10, seed=6)
    for _, game in df.groupby("game_id"):
        white_rows = game[game["side"] == "w"]["result"].unique()
        black_rows = game[game["side"] == "b"]["result"].unique()
        assert len(white_rows) <= 1
        assert len(black_rows) <= 1
        if len(white_rows) and len(black_rows):
            assert white_rows[0] == 1.0 - black_rows[0]
