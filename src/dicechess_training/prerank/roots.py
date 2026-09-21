"""Choose the positions the pre-ranker's corpus will be built from.

Labelling one root costs the champion's expensive evaluation on every turn the rules allow — around
120 of them, each a pair of 216-outcome king-capture searches. So which roots to visit is a
decision with a price, and it is made here rather than inside the generator, where it would be
buried under the work it causes.

Three properties this module exists to give the choice.

**One root, once.** A group's identity is its position and its roll, so the same pair twice is the
same list twice, which `load_groups` refuses. Repeats are ordinary rather than exotic — measured on
the committed sample, 11% of rows repeat a root and 704 roots occur in more than one game — so
deduplication happens here, deterministically, and the occurrence that survives decides which game
the group is split by.

**A sample that grows without invalidating what came before.** Roots are ordered by a hash of their
own identity and the first `limit` are taken, so raising the limit adds roots without disturbing
the ones already chosen. A corpus built from 5,000 roots is a byte-prefix of the one built from
10,000, which means a bigger corpus reuses rather than replaces the core-hours already spent.

**A reproduction record.** The roots file *is* the producer's immutable input — the manifest's
`source_sha256` binds it — so beside it goes a record of where it came from: the shards, their
content digests, the rule, and what the selection covers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from dicechess_training import schema
from dicechess_training.benchmark.splits import split_for
from dicechess_training.prerank.groups import group_id_for, root_key

#: Header of the roots file. Tab-separated, because a FEN contains spaces but never a tab, and
#: because the generator that reads it must not have to agree with this repository about JSON.
HEADER = ("game_id", "fen", "dice", "side")

ROOTS_FILE = "roots.tsv"
PROVENANCE_FILE = "roots-provenance.json"

#: The ordering label. Fixed rather than a parameter: two samples drawn under different labels are
#: not comparable and not nested, and nothing is gained that a different `limit` does not give.
ORDER_LABEL = "prerank-roots-v1"

SELECTION_RULE = f'sha256("{ORDER_LABEL}:" + canonical_fen + "|" + dice) ascending, first `limit`'

#: What one root turned out to cost, measured on 400 roots of the committed sample at engine
#: 0.12.0: 120.7 candidates per root read, 235 bytes of JSON each, 0.60 ms per candidate on eight
#: threads. Used only to project the bill before it is run up — the corpus is cheap in CPU and
#: expensive in JSON, and 43,692 roots is 1.2 GB, which the loader would have to hold as objects.
CANDIDATES_PER_ROOT = 120.7
BYTES_PER_CANDIDATE = 235
MILLISECONDS_PER_CANDIDATE = 0.6


def projected_cost(roots: int) -> dict:
    """What labelling this many roots is likely to cost, from what the last measurement showed."""
    candidates = roots * CANDIDATES_PER_ROOT
    return {
        "candidates": int(candidates),
        "megabytes": round(candidates * BYTES_PER_CANDIDATE / 1e6, 1),
        "core_hours": round(candidates * MILLISECONDS_PER_CANDIDATE * 8 / 1000 / 3600, 2),
    }


class RootsError(ValueError):
    """A roots selection was refused. The message says what, never where."""


def order_key(fen: str, dice: str) -> str:
    """Where a root falls in the selection order. A function of the root alone, so the order is
    the same whatever shards it was found in and whatever order they were read."""
    canonical, roll = root_key({"root_fen": fen, "dice": dice})
    return hashlib.sha256(f"{ORDER_LABEL}:{canonical}|{roll}".encode()).hexdigest()


def distinct_roots(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per position-and-roll, attributed to its earliest occurrence.

    Earliest by `(game_id, ply)` rather than by file order, because file order depends on how the
    shards were written and the attribution decides the split. A root that occurs in six games is
    one decision; which of the six it is charged to has to be a rule, not an accident.
    """
    for column in ("game_id", "ply", "fen", "dice", "side"):
        if column not in frame.columns:
            raise RootsError(f"the source is missing {column}")
    if frame.empty:
        raise RootsError("the source has no rows")

    declared = frame["fen"].astype(str).str.split().str[1]
    mismatched = int((declared != frame["side"].astype(str)).sum())
    if mismatched:
        raise RootsError(f"{mismatched} rows declare a side that is not the FEN's side to move")

    chosen = (
        frame.assign(
            _key=[
                order_key(fen, dice) for fen, dice in zip(frame["fen"], frame["dice"], strict=True)
            ]
        )
        .sort_values(["_key", "game_id", "ply"], kind="stable")
        .drop_duplicates("_key", keep="first")
    )
    return chosen.reset_index(drop=True)


def select(frame: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """The roots to label, in the order the corpus will hold them."""
    if limit is not None and limit <= 0:
        raise RootsError("limit must be positive")
    roots = distinct_roots(frame)
    return roots if limit is None else roots.head(limit)


def write_roots(roots: pd.DataFrame, path: Path) -> str:
    """Write the roots file and return its digest, which becomes the dataset's `source_sha256`."""
    lines = ["\t".join(HEADER)]
    for row in roots.itertuples():
        for field in (row.game_id, row.fen, row.dice, row.side):
            if "\t" in str(field) or "\n" in str(field):
                raise RootsError("a root field contains a tab or newline")
        lines.append(f"{row.game_id}\t{row.fen}\t{row.dice}\t{row.side}")
    payload = "\n".join(lines) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def coverage(roots: pd.DataFrame) -> dict:
    """What the selection covers, in the terms the reader of a corpus will ask about."""
    splits: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for game in roots["game_id"]:
        splits[split_for(str(game))] += 1
    plies = sorted(int(ply) for ply in roots["ply"])
    sides = roots["side"].value_counts().to_dict()
    return {
        "roots": len(roots),
        "games": int(roots["game_id"].nunique()),
        "splits": splits,
        "sides": {str(side): int(count) for side, count in sides.items()},
        "ply": {
            "min": plies[0],
            "median": plies[len(plies) // 2],
            "p95": plies[int(len(plies) * 0.95)],
            "max": plies[-1],
        },
    }


def export(source: Path, destination: Path, limit: int | None = None) -> dict:
    """Read the shards, choose the roots, and write the file plus its reproduction record."""
    if limit is not None and limit <= 0:
        raise RootsError("limit must be positive")
    frame = schema.read_shards(str(source))
    available = distinct_roots(frame)
    roots = available if limit is None else available.head(limit)
    check_group_ids(roots)
    destination.mkdir(parents=True, exist_ok=True)
    digest = write_roots(roots, destination / ROOTS_FILE)

    record = {
        "generator": "dicechess_training.prerank.roots",
        "selection": SELECTION_RULE,
        "limit": limit,
        "shards": [
            {"file": path.name, "content_sha256": schema.shard_content_digest(path)}
            for path in sorted(Path(source).glob("*.parquet"))
        ],
        "rows_read": int(len(frame)),
        "distinct_roots": int(len(available)),
        "roots_sha256": digest,
        **coverage(roots),
    }
    (destination / PROVENANCE_FILE).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def check_group_ids(roots: pd.DataFrame) -> None:
    """Refuse a selection two of whose roots would carry the same group id.

    The id is a 64-bit prefix of a digest, so this cannot happen by repetition — repetition is
    already gone — only by collision. It is checked anyway because the failure it would otherwise
    cause is a whole corpus refused on admission with nothing to point at.
    """
    ids = [group_id_for(fen, dice) for fen, dice in zip(roots["fen"], roots["dice"], strict=True)]
    if len(set(ids)) != len(ids):
        raise RootsError("two distinct roots collide on one group id")
