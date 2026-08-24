"""Toy data generation and game-level splitting.

The synthetic generator exists so the demo and tests run before the real
bot-vs-bot sample lands (issue #4). Its positions are random piece placements —
NOT legal game states — which is fine for validating the pipeline plumbing and
meaningless for playing strength.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import PIECE_TYPES, RESULTS, validate_frame

_NON_KING = PIECE_TYPES.replace("K", "")


def _random_fen(rng: np.random.Generator) -> str:
    """Random sparse placement with exactly two kings. Not necessarily legal."""
    board = np.full(64, "", dtype=object)
    squares = rng.choice(64, size=2 + int(rng.integers(2, 12)), replace=False)
    board[squares[0]] = "K"
    board[squares[1]] = "k"
    for sq in squares[2:]:
        piece = _NON_KING[int(rng.integers(len(_NON_KING)))]
        board[sq] = piece if rng.random() < 0.5 else piece.lower()
    ranks = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            piece = board[rank * 8 + file]
            if piece:
                row += (str(empty) if empty else "") + piece
                empty = 0
            else:
                empty += 1
        ranks.append(row + (str(empty) if empty else ""))
    return "/".join(ranks)


def generate_toy_games(n_games: int, seed: int = 0) -> pd.DataFrame:
    """Generate a schema-v0 DataFrame of random games (positions share an outcome)."""
    rng = np.random.default_rng(seed)
    rows = []
    for game in range(n_games):
        outcome_white = RESULTS[int(rng.integers(3))]
        for ply in range(int(rng.integers(8, 32))):
            side = "w" if ply % 2 == 0 else "b"
            result = outcome_white if side == "w" else 1.0 - outcome_white
            rows.append(
                {
                    "game_id": f"toy-{seed}-{game}",
                    "ply": ply,
                    "fen": _random_fen(rng),
                    "dice": "".join(PIECE_TYPES[int(i)] for i in rng.integers(6, size=3)),
                    "side": side,
                    "result": result,
                }
            )
    df = pd.DataFrame(rows)
    validate_frame(df)
    return df


def split_by_game(
    df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by GAME, not by position.

    Positions within a game share a label and near-duplicate context; a
    position-level split leaks between train and validation and inflates
    validation metrics (measured on the private pipeline).
    """
    games = np.asarray(df["game_id"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_fraction))
    val_games = set(games[:n_val])
    val_mask = df["game_id"].isin(val_games)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)
