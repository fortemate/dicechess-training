"""What the grouped dataset must be, stated as refusals.

Each test names a way a pre-ranker corpus can be wrong and asserts the loader refuses it. The
positive test is one: a dataset built the way the generator will build it loads, and its splits
move whole lists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.contracts import kcp13
from dicechess_training.prerank import (
    GroupsError,
    assign_splits,
    duplicate_roots,
    group_id_for,
    load_groups,
)

ENGINE_VERSION = "0.12.0"
SCHEMA = "rich-9-v1"
GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "rich9"
    / "golden-engine-0.12.0.json"
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
OTHER = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -"


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _candidate(result: str, target: float, columns: int) -> dict:
    return {
        "moves": ["e2e4"],
        "result_fen": result,
        "features": [0.0] * columns,
        "target": target,
    }


def _group(root: str = START, dice: str = "BNP", game: str = "game-1", side: str = "w") -> dict:
    columns = len(_golden()["columns"])
    return {
        "group_id": group_id_for(root, dice),
        "game_id": game,
        "root_fen": root,
        "dice": dice,
        "side": side,
        "candidates": [
            _candidate("position-a", 120.0, columns),
            _candidate("position-b", -35.5, columns),
        ],
    }


def _write(directory: Path, payload: list[dict], **manifest_overrides) -> Path:
    """`payload`, not `groups`: the manifest has a `groups` key, and a keyword override for it must
    not collide with this helper's own parameter."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "license.txt").write_text("Owner-controlled. Not for redistribution.\n", "utf-8")
    (directory / "groups.json").write_text(json.dumps(payload, indent=1) + "\n", "utf-8")
    manifest = {
        "schema": "playground-prerank-groups-v1",
        "version": "1.0.0",
        "perspective": "side-to-move",
        "target": "expensive-candidate-evaluation",
        "kind": "owner-controlled",
        "license": "Fortemate Owner-Controlled Data Policy v1",
        "license_evidence_sha256": kcp13.sha256_of(directory / "license.txt"),
        "groups_sha256": kcp13.sha256_of(directory / "groups.json"),
        "source_sha256": "a" * 64,
        "golden_sha256": benchmark_core.digest(_golden()),
        "teacher": {"id": "champion-rescore-v1", "weights_sha256": "b" * 64},
        "feature_schema": SCHEMA,
        "columns": _golden()["columns"],
        "engine_version": ENGINE_VERSION,
        "groups": len(payload),
        "candidates": sum(len(group["candidates"]) for group in payload),
    }
    manifest.update(manifest_overrides)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    return directory


def test_a_well_formed_dataset_loads(tmp_path: Path) -> None:
    manifest, groups = load_groups(_write(tmp_path / "ds", [_group()]))
    assert manifest["feature_schema"] == SCHEMA
    assert len(groups) == 1
    assert len(groups[0]["candidates"]) == 2


def test_splits_move_whole_lists(tmp_path: Path) -> None:
    groups = [_group(game=f"game-{i}") for i in range(40)]
    _, loaded = load_groups(_write(tmp_path / "ds", _distinct_roots(groups)))
    splits = assign_splits(loaded)
    assert len(splits) == len(loaded)
    assert set(splits) <= {"train", "validation", "test"}
    # One split per group is the whole point: a list cannot be half in and half out.
    assert all(isinstance(split, str) for split in splits)


def _distinct_roots(groups: list[dict]) -> list[dict]:
    """Same position, different games — but the loader refuses repeated lists, so vary the dice."""
    faces = ["BNP", "BKN", "KNQ", "NPR", "PQR", "BNQ", "KPR", "NPP", "BBN", "KQR"]
    out = []
    for index, group in enumerate(groups):
        dice = faces[index % len(faces)]
        root = START if index % 2 == 0 else OTHER
        side = "w" if index % 2 == 0 else "b"
        fresh = _group(root=root, dice=dice, game=group["game_id"], side=side)
        out.append(fresh)
    return _deduplicate(out)


def _deduplicate(groups: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for group in groups:
        if group["group_id"] in seen:
            continue
        seen.add(group["group_id"])
        unique.append(group)
    return unique


def test_the_same_root_in_two_splits_is_reported(tmp_path: Path) -> None:
    """Games are disjoint; positions are not. Two games reaching the same root under the same roll
    is the leak that a game-level split cannot see."""
    groups = [_group(game="game-a"), _group(game="game-b")]
    # Both groups are the same list, so give the second a different id the loader will accept.
    groups[1]["game_id"] = "game-b"
    splits = ["train", "test"]
    found = duplicate_roots(groups, splits)
    assert len(found) == 1
    assert next(iter(found.values())) == {"train", "test"}


def test_a_root_in_one_split_is_not_reported() -> None:
    groups = [_group(game="game-a"), _group(game="game-b")]
    assert duplicate_roots(groups, ["train", "train"]) == {}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"schema": "playground-rows-v1"}, "unsupported dataset schema"),
        ({"version": "1.0"}, "invalid dataset version"),
        ({"perspective": "white"}, "wrong perspective"),
        ({"target": "decisive-game-outcome"}, "unsupported target"),
        ({"kind": "unknown"}, "unverified dataset origin"),
        ({"license": ""}, "missing data terms"),
        ({"groups_sha256": "not-a-digest"}, "groups_sha256 is missing or unusable"),
        ({"groups_sha256": "c" * 64}, "dataset digest mismatch"),
        ({"feature_schema": "kcp-13"}, "unsupported feature schema for a pre-ranker"),
        ({"engine_version": "9.9.9"}, "no committed golden corpus"),
        ({"columns": ["wrong"]}, "declared columns are not the engine's layout"),
        ({"golden_sha256": "d" * 64}, "golden corpus digest mismatch"),
        ({"teacher": {"id": "x"}}, "teacher's weights digest is unusable"),
        ({"teacher": {"weights_sha256": "b" * 64}}, "the teacher has no id"),
        ({"groups": 7}, "group count"),
        ({"candidates": 99}, "candidate count"),
    ],
)
def test_a_manifest_that_lies_is_refused(tmp_path: Path, override: dict, message: str) -> None:
    directory = _write(tmp_path / "ds", [_group()], **override)
    with pytest.raises(GroupsError) as error:
        load_groups(directory)
    assert message in str(error.value)


def _mutate(group: dict, **changes) -> dict:
    mutated = json.loads(json.dumps(group))
    mutated.update(changes)
    return mutated


@pytest.mark.parametrize(
    ("group", "message"),
    [
        (_mutate(_group(), dice="PNB"), "canonical, sorted piece letters"),
        (_mutate(_group(), dice="BN"), "canonical, sorted piece letters"),
        (_mutate(_group(), dice="BNX"), "canonical, sorted piece letters"),
        (_mutate(_group(), side="b"), "not the root's side to move"),
        (_mutate(_group(), group_id="0" * 16), "does not identify its own root"),
        (_mutate(_group(), game_id=" "), "has no game"),
        (_mutate(_group(), candidates=[]), "has no candidates"),
    ],
)
def test_a_group_that_breaks_its_own_shape_is_refused(
    tmp_path: Path, group: dict, message: str
) -> None:
    with pytest.raises(GroupsError) as error:
        load_groups(_write(tmp_path / "ds", [group]))
    assert message in str(error.value)


def test_transpositions_must_already_be_collapsed(tmp_path: Path) -> None:
    group = _group()
    group["candidates"][1]["result_fen"] = group["candidates"][0]["result_fen"]
    with pytest.raises(GroupsError, match="reach the same position"):
        load_groups(_write(tmp_path / "ds", [group]))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"features": [0.0]}, "do not match the declared columns"),
        ({"features": [float("nan")] * 9}, "not finite"),
        ({"features": [True] * 9}, "not a number"),
        ({"target": float("inf")}, "not finite"),
        ({"target": "high"}, "not a number"),
        ({"moves": []}, "has no moves"),
    ],
)
def test_a_candidate_that_breaks_its_own_shape_is_refused(
    tmp_path: Path, mutation: dict, message: str
) -> None:
    group = _group()
    group["candidates"][0].update(mutation)
    with pytest.raises(GroupsError) as error:
        load_groups(_write(tmp_path / "ds", [group]))
    assert message in str(error.value)


def test_the_same_list_twice_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GroupsError, match="appears twice"):
        load_groups(_write(tmp_path / "ds", [_group(), _group(game="game-2")]))
