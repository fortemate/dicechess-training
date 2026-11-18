import numpy as np
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


def test_random_fen_preserves_seeded_sequence_and_following_rng_draws():
    rng = np.random.default_rng(42)
    actual = [data._random_fen(rng) for _ in range(3)]
    assert actual == [
        "7p/8/k6n/8/3K4/8/8/8",
        "1p6/1k6/1Br4R/8/2q1nK2/6q1/3R4/7q",
        "8/2p5/3q3R/8/3rQ3/1k3rr1/K2N4/4Q3",
    ]
    np.testing.assert_array_equal(rng.integers(0, 1_000_000, size=3), [555646, 458915, 505606])


@pytest.mark.parametrize("seed", range(12))
def test_random_fen_has_two_kings_and_eight_complete_ranks(seed):
    rng = np.random.default_rng(seed)
    for _ in range(3):
        fen = data._random_fen(rng)
        assert fen.count("K") == 1
        assert fen.count("k") == 1
        ranks = fen.split("/")
        assert len(ranks) == 8
        for rank in ranks:
            assert sum(int(ch) if ch.isdigit() else 1 for ch in rank) == 8
