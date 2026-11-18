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
    for empty in ("", "   "):
        with pytest.raises(ValueError, match="empty"):
            features.parse_fen_placement(empty)


@pytest.mark.parametrize("suffix", ["", " w - - 0 1", " ignored metadata"])
def test_placement_maps_all_piece_planes_and_ignores_other_fields(suffix):
    fen = f" \tPNBRQK2/8/8/8/8/8/8/pnbrqk2{suffix}\n"
    actual = features.parse_fen_placement(fen)
    expected = np.zeros((12, 64), dtype=np.float32)
    for piece in range(6):
        expected[piece, 56 + piece] = 1.0
        expected[6 + piece, piece] = 1.0

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("rank", ["8", "08", "11111111", "٨"])
def test_placement_preserves_accepted_numeric_rank_spellings(rank):
    fen = "/".join([rank] * 8)
    actual = features.parse_fen_placement(fen)
    np.testing.assert_array_equal(actual, np.zeros((12, 64), dtype=np.float32))


@pytest.mark.parametrize(
    ("fen", "error", "message"),
    [
        ("", ValueError, "FEN is empty: ''"),
        ("   ", ValueError, "FEN is empty: '   '"),
        (
            "8/8/8/8/8/8/8",
            ValueError,
            "FEN placement must have 8 ranks: '8/8/8/8/8/8/8'",
        ),
        (
            "8/8/8/8/8/8/8/X7",
            ValueError,
            "unexpected piece 'X' in FEN '8/8/8/8/8/8/8/X7'",
        ),
        (
            "8/8/8/8/8/8/8/7",
            ValueError,
            "rank '7' does not span 8 files in FEN '8/8/8/8/8/8/8/7'",
        ),
        (
            "9/8/8/8/8/8/8/8",
            ValueError,
            "rank '9' does not span 8 files in FEN '9/8/8/8/8/8/8/8'",
        ),
        (
            "8P/8/8/8/8/8/8/8",
            IndexError,
            "index 64 is out of bounds for axis 1 with size 64",
        ),
        ("²/8/8/8/8/8/8/8", ValueError, "invalid literal for int() with base 10: '²'"),
    ],
)
def test_placement_preserves_error_type_message_and_order(fen, error, message):
    with pytest.raises(error) as caught:
        features.parse_fen_placement(fen)
    assert type(caught.value) is error
    assert str(caught.value) == message
