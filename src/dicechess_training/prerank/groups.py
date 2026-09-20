"""The `playground-prerank-groups-v1` contract: what a pre-ranker's training data must be.

Three properties this file exists to enforce, in the order they matter.

**Lists stay lists.** A pre-ranker is judged on the order it puts one root's candidates in, so the
data is a list of groups and never a table of rows with a group column. A flat table can be
filtered, sorted or shuffled into something that still loads and no longer has lists in it; this
shape cannot.

**The split is by game, the audit is by position.** Assigning candidates rather than groups would
put a root's own alternatives on both sides of the split, which is leakage of the crudest kind, so
splitting takes whole groups by `game_id`. That alone is not enough: the same position with the
same dice occurs in many games, and two such groups landing in different splits is the same leak
wearing a different coat. `duplicate_roots` finds those; it does not decide what to do about them.

**Features come from the engine, not from here.** The manifest names a feature schema, and the
columns are checked against the committed golden corpus for that schema and engine version rather
than against a list retyped in Python. Nothing in this module can invent a feature layout.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.benchmark.splits import split_for

SCHEMA_ID = "playground-prerank-groups-v1"
MANIFEST_FILE = "manifest.json"
GROUPS_FILE = "groups.json"
LICENSE_FILE = "license.txt"

PERSPECTIVE = "side-to-move"
TARGET = "expensive-candidate-evaluation"
KINDS = ("synthetic", "public-licensed", "owner-controlled")
DICE_FACES = "BKNPQR"

#: Feature schemas a pre-ranker may declare. The seam scores every legal turn before the search's
#: deadline is consulted, so a schema whose cost follows the branching factor — `kcp-13` and
#: anything richer — cannot be served here whatever its accuracy.
STUDENT_SCHEMAS = {"rich-9-v1": "rich9"}

_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_ENGINE_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


class GroupsError(ValueError):
    """A grouped dataset was refused. The message says what, never where."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise GroupsError(message)


def group_id_for(root_fen: str, dice: str) -> str:
    """The list's identity: the position it starts from and the roll that constrains it.

    Derived rather than supplied so two generators cannot disagree about what one list is, and
    truncated because it is an index, not evidence — the payload's digest is the evidence.
    """
    canonical = _canonical_fen(root_fen)
    return hashlib.sha256(f"prerank-v1:{canonical}|{dice}".encode()).hexdigest()[:16]


def _canonical_fen(fen: str) -> str:
    fields = fen.split()
    _require(len(fields) >= 4, "root position is not a four-field FEN")
    return " ".join([fields[0], fields[1], "".join(sorted(fields[2])), fields[3]])


def root_key(group: dict) -> tuple[str, str]:
    """What makes two groups the same list of decisions: the position and the roll."""
    return _canonical_fen(group["root_fen"]), group["dice"]


def _golden_columns(schema_id: str, engine_version: str) -> tuple[tuple[str, ...], str]:
    _require(schema_id in STUDENT_SCHEMAS, "unsupported feature schema for a pre-ranker")
    _require(
        _ENGINE_VERSION.match(engine_version) is not None,
        "invalid engine version",
    )
    path = (
        benchmark_core.ROOT
        / "tests"
        / "fixtures"
        / STUDENT_SCHEMAS[schema_id]
        / f"golden-engine-{engine_version}.json"
    )
    _require(path.is_file(), "no committed golden corpus for that schema and engine version")
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(raw.get("schema") == schema_id, "golden corpus declares another schema")
    columns = tuple(raw.get("columns", ()))
    _require(bool(columns), "golden corpus declares no columns")
    return columns, benchmark_core.digest(raw)


def _check_manifest(manifest: dict, directory: Path) -> tuple[tuple[str, ...], dict]:
    _require(manifest.get("schema") == SCHEMA_ID, "unsupported dataset schema")
    _require(
        _VERSION.match(str(manifest.get("version", ""))) is not None, "invalid dataset version"
    )
    _require(manifest.get("perspective") == PERSPECTIVE, "wrong perspective")
    _require(manifest.get("target") == TARGET, "unsupported target")
    _require(manifest.get("kind") in KINDS, "unverified dataset origin")
    _require(
        str(manifest.get("license", "")) not in ("", "unknown", "unresolved"), "missing data terms"
    )
    for key in ("groups_sha256", "license_evidence_sha256", "source_sha256", "golden_sha256"):
        try:
            benchmark_core.require_sha(manifest.get(key))
        except ValueError as error:
            raise GroupsError(f"{key} is missing or unusable") from error

    from dicechess_training.contracts import kcp13

    _require(
        kcp13.sha256_of(directory / LICENSE_FILE) == manifest["license_evidence_sha256"],
        "data terms digest mismatch",
    )
    _require(
        kcp13.sha256_of(directory / GROUPS_FILE) == manifest["groups_sha256"],
        "dataset digest mismatch",
    )

    # The teacher is private and stays private: the dataset carries which evaluation produced the
    # targets and the digest of the weights it ran with, never the weights. Without that pair a
    # target is a number nobody can attribute, and two datasets built a month apart become
    # indistinguishable.
    teacher = manifest.get("teacher")
    _require(isinstance(teacher, dict), "the dataset does not say which teacher scored it")
    _require(str(teacher.get("id", "")).strip() != "", "the teacher has no id")
    try:
        benchmark_core.require_sha(teacher.get("weights_sha256"))
    except ValueError as error:
        raise GroupsError("the teacher's weights digest is unusable") from error

    columns, golden_digest = _golden_columns(
        str(manifest.get("feature_schema")), str(manifest.get("engine_version"))
    )
    _require(
        tuple(manifest.get("columns", ())) == columns,
        "declared columns are not the engine's layout for that schema",
    )
    _require(manifest["golden_sha256"] == golden_digest, "golden corpus digest mismatch")
    return columns, manifest


def _check_group(group: dict, columns: tuple[str, ...]) -> None:
    for key in ("group_id", "game_id", "root_fen", "dice", "side", "candidates"):
        _require(key in group, f"a group is missing {key}")
    _require(str(group["game_id"]).strip() != "", "a group has no game")

    dice = group["dice"]
    _require(
        isinstance(dice, str)
        and len(dice) == 3
        and all(face in DICE_FACES for face in dice)
        and list(dice) == sorted(dice),
        "dice must be three canonical, sorted piece letters",
    )
    _require(group["side"] in ("w", "b"), "a group has no side to move")
    _require(
        _canonical_fen(group["root_fen"]).split()[1] == group["side"],
        "the declared side is not the root's side to move",
    )
    _require(
        group["group_id"] == group_id_for(group["root_fen"], dice),
        "group_id does not identify its own root and roll",
    )

    candidates = group["candidates"]
    _require(isinstance(candidates, list) and candidates, "a group has no candidates")
    seen_results: set[str] = set()
    for candidate in candidates:
        for key in ("moves", "result_fen", "features", "target"):
            _require(key in candidate, f"a candidate is missing {key}")
        moves = candidate["moves"]
        _require(isinstance(moves, list) and moves, "a candidate has no moves")
        _require(all(isinstance(move, str) and move for move in moves), "a move is not a string")

        result = candidate["result_fen"]
        _require(isinstance(result, str) and result.strip() != "", "a candidate has no result")
        # Transpositions are one decision, not several: the same resulting position reached by two
        # move orders is one candidate, and leaving both in would let a ranker be scored twice for
        # the same choice.
        _require(result not in seen_results, "two candidates reach the same position")
        seen_results.add(result)

        features = candidate["features"]
        _require(
            isinstance(features, list) and len(features) == len(columns),
            "a candidate's features do not match the declared columns",
        )
        _require(
            all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in features
            ),
            "a feature is not a number",
        )
        _require(
            all(value == value and abs(value) != float("inf") for value in features),
            "a feature is not finite",
        )

        target = candidate["target"]
        _require(
            isinstance(target, (int, float)) and not isinstance(target, bool),
            "a target is not a number",
        )
        _require(target == target and abs(target) != float("inf"), "a target is not finite")


def load_groups(directory: str | Path) -> tuple[dict, list[dict]]:
    """Admit a grouped dataset, or refuse it. Returns the manifest and its groups."""
    directory = Path(directory)
    manifest = benchmark_core.read_json(directory / MANIFEST_FILE)
    columns, manifest = _check_manifest(manifest, directory)

    groups = json.loads((directory / GROUPS_FILE).read_text(encoding="utf-8"))
    _require(isinstance(groups, list) and groups, "the dataset has no groups")
    for group in groups:
        _check_group(group, columns)

    ids = [group["group_id"] for group in groups]
    _require(len(set(ids)) == len(ids), "the same list appears twice")
    _require(
        manifest.get("groups") == len(groups),
        "the manifest's group count is not the number of groups",
    )
    _require(
        manifest.get("candidates") == sum(len(group["candidates"]) for group in groups),
        "the manifest's candidate count is not the number of candidates",
    )
    return manifest, groups


def assign_splits(groups: list[dict]) -> list[str]:
    """One split per group, decided by the game it came from.

    Whole groups move together because a root's candidates are the thing being ordered; splitting
    inside a list would train on part of an answer and test on the rest of it. The rule is the
    benchmark's own, so a game that is train for the evaluator is train here too.
    """
    return [split_for(str(group["game_id"])) for group in groups]


def duplicate_roots(groups: list[dict], splits: list[str]) -> dict[tuple[str, str], set[str]]:
    """Roots that occur in more than one split, with the splits they occur in.

    Games are disjoint by construction; positions are not. The same opening position under the same
    roll is reached by many games, and when two of them fall on opposite sides of the split the
    model is tested on a list it has already been fitted to. Reported rather than dropped: whether
    to exclude, keep or rebuild is the owner's call, and silently deleting rows would hide how much
    of the corpus is affected.
    """
    seen: dict[tuple[str, str], set[str]] = {}
    for group, split in zip(groups, splits, strict=True):
        seen.setdefault(root_key(group), set()).add(split)
    return {key: splits_seen for key, splits_seen in seen.items() if len(splits_seen) > 1}
