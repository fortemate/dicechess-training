"""Feature encoding v0: rawboard-768 + dice piece-type counts = 774 floats.

Everything is canonicalized to the MOVER's perspective (own minus opponent,
never white minus black): planes 0-5 are the side to move's pieces, planes
6-11 the opponent's, and for Black the board is rank-mirrored so the mover's
back rank is always rank 0. This makes one network serve both colors and is
the same convention the private pipeline settled on.
"""

from __future__ import annotations

import numpy as np

from .schema import DICE_COUNT, PIECE_TYPES, validate_dice

BOARD_DIM = 768  # 12 planes x 64 squares
DICE_DIM = len(PIECE_TYPES)
FEATURE_DIM = BOARD_DIM + DICE_DIM

_PIECE_INDEX = {p: i for i, p in enumerate(PIECE_TYPES)}


def parse_fen_placement(fen: str) -> np.ndarray:
    """Parse the placement field of a FEN into a (12, 64) white-perspective array.

    Planes 0-5: white P N B R Q K; planes 6-11: black. Square index is
    rank * 8 + file with rank 0 = rank '1' (White's back rank).
    """
    fields = fen.split()
    if not fields:
        raise ValueError(f"FEN is empty: {fen!r}")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN placement must have 8 ranks: {fen!r}")
    planes = np.zeros((12, 64), dtype=np.float32)
    for fen_row, rank_str in enumerate(ranks):
        rank = 7 - fen_row  # FEN lists rank 8 first
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
                continue
            upper = ch.upper()
            if upper not in _PIECE_INDEX:
                raise ValueError(f"unexpected piece {ch!r} in FEN {fen!r}")
            plane = _PIECE_INDEX[upper] + (0 if ch.isupper() else 6)
            planes[plane, rank * 8 + file] = 1.0
            file += 1
        if file != 8:
            raise ValueError(f"rank {rank_str!r} does not span 8 files in FEN {fen!r}")
    return planes


def _to_mover_perspective(planes: np.ndarray, side: str) -> np.ndarray:
    """Reorder planes to own/opponent and rank-mirror the board for Black."""
    if side == "w":
        return planes
    mirrored = planes.reshape(12, 8, 8)[:, ::-1, :].reshape(12, 64)
    return np.concatenate([mirrored[6:], mirrored[:6]])


def encode_dice(dice: str) -> np.ndarray:
    """Encode the dice multiset as per-piece-type counts scaled to [0, 1]."""
    counts = np.zeros(DICE_DIM, dtype=np.float32)
    for ch in validate_dice(dice):
        counts[_PIECE_INDEX[ch]] += 1.0
    return counts / DICE_COUNT


def encode_features(fen: str, dice: str, side: str) -> np.ndarray:
    """Encode one position into the 774-float feature vector (mover perspective)."""
    if side not in ("w", "b"):
        raise ValueError(f"side must be 'w' or 'b', got {side!r}")
    board = _to_mover_perspective(parse_fen_placement(fen), side)
    return np.concatenate([board.reshape(BOARD_DIM), encode_dice(dice)])


def encode_frame(df) -> np.ndarray:
    """Encode a schema-v0 DataFrame into an (n, 774) float32 matrix."""
    columns = zip(df["fen"], df["dice"], df["side"], strict=True)
    return np.stack([encode_features(fen, dice, side) for fen, dice, side in columns])
