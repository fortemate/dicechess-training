import numpy as np
import pytest

from dicechess_training import features


def _mirror_and_swap(fen: str) -> str:
    """Rank-mirror the placement and swap piece colors (the black twin position)."""
    return "/".join(rank.swapcase() for rank in reversed(fen.split()[0].split("/")))


def test_feature_dimensions():
    vec = features.encode_features("4k3/8/8/8/8/8/4P3/4K3", "NQK", "w")
    assert vec.shape == (features.FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_dice_counts():
    dice = features.encode_dice("NNQ")
    assert dice.sum() == pytest.approx(1.0)
    assert dice[features._PIECE_INDEX["N"]] == pytest.approx(2 / 3)
    assert dice[features._PIECE_INDEX["Q"]] == pytest.approx(1 / 3)


def test_mover_perspective_symmetry():
    """White's view of a position == Black's view of its color-swapped mirror."""
    fen = "r3k3/1pp5/8/8/3Q4/8/PP6/4K2R"
    white_view = features.encode_features(fen, "RNB", "w")
    black_view = features.encode_features(_mirror_and_swap(fen), "RNB", "b")
    np.testing.assert_array_equal(white_view, black_view)


def test_piece_counts_preserved():
    fen = "4k3/2q5/8/8/8/8/2Q5/4K3"
    for side in ("w", "b"):
        vec = features.encode_features(fen, "QQQ", side)
        assert vec[: features.BOARD_DIM].sum() == 4  # two kings, two queens


def test_invalid_fen_rejected():
    with pytest.raises(ValueError):
        features.parse_fen_placement("4k3/8/8")
    with pytest.raises(ValueError):
        features.parse_fen_placement("4k4/8/8/8/8/8/8/4K3")
