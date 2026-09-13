"""Stable group assignment and conservative dice-free exact-position overlap accounting."""

import hashlib


def split_for(group: str) -> str:
    value = int(hashlib.sha256(("playground-v1:" + group).encode()).hexdigest(), 16) % 10000
    if value < 8000:
        return "train"
    return "validation" if value < 9000 else "test"


def position_key(row):
    # Enrichment must supply engine-canonical FEN. Clocks and roll are outside kcp-13.
    fields = row["fen"].split()
    return " ".join([fields[0], fields[1], "".join(sorted(fields[2])), fields[3]])


def assignments(rows):
    """A declared group may contain several games/roots; one game/root cannot cross groups."""
    owners = {}
    for row in rows:
        for field in ("game_id", "root_id"):
            if row.get(field):
                key = field, row[field]
                if key in owners and owners[key] != row["group_id"]:
                    raise ValueError("game or root crosses declared groups")
                owners[key] = row["group_id"]
    return [split_for(row["group_id"]) for row in rows]


def leakage(rows, splits):
    keys = {
        s: {position_key(r) for r, part in zip(rows, splits, strict=True) if part == s}
        for s in ("train", "validation", "test")
    }
    return {
        f"{a}:{b}": len(keys[a] & keys[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
