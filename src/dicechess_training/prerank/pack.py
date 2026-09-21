"""Turn a generated corpus into an admissible dataset by writing its manifest.

The generator lives in another repository, because the teacher it labels with is that bot's own
evaluation. It writes `groups.json` — the part only it can produce — and `generation.json`, a
record of the run. It deliberately does not write `manifest.json`, and this module is why.

A manifest has to state the canonical-JSON digest of a golden corpus committed *here*, under a
convention defined *here*, checked on admission by code *here*. Restating that convention in Scala
would put a digest format in two places, which is worse than a contract in two places: the two
would agree until the day a float was formatted differently, and the symptom would be every corpus
refused with no indication why.

So the split follows what each side can actually know. The generator states the engine it ran, the
teacher it used and the digest of its input; this reads that record, computes the digests the
contract defines, writes the manifest, and then — the point of doing it here — **loads the result
through `load_groups`**. A manifest this module writes cannot be one the loader rejects, because
the loader has already accepted it before the command returns.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.contracts import kcp13
from dicechess_training.prerank.groups import (
    GROUPS_FILE,
    KINDS,
    LICENSE_FILE,
    MANIFEST_FILE,
    PERSPECTIVE,
    SCHEMA_ID,
    TARGET,
    assign_splits,
    golden_columns,
    load_groups,
)

GENERATION_FILE = "generation.json"

DEFAULT_VERSION = "0.1.0"
DEFAULT_KIND = "owner-controlled"


class PackError(ValueError):
    """A corpus could not be packed. The message says what, never where."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise PackError(message)


def read_generation(directory: Path) -> dict:
    """The generator's record, checked for the fields a manifest is built out of.

    Checked rather than trusted: this file crosses a repository boundary, and a missing teacher or
    a digest in the wrong shape should be named here, not surface later as a manifest the loader
    rejects for a reason that no longer points at the cause.
    """
    path = directory / GENERATION_FILE
    _require(path.is_file(), "the corpus has no generation record")
    record = benchmark_core.read_json(path)

    _require(record.get("schema") == SCHEMA_ID, "the generator produced another dataset schema")
    for key in ("feature_schema", "engine_version", "roots_sha256"):
        _require(str(record.get(key, "")).strip() != "", f"the generation record has no {key}")

    teacher = record.get("teacher")
    _require(isinstance(teacher, dict), "the generation record names no teacher")
    _require(str(teacher.get("id", "")).strip() != "", "the teacher has no id")
    for key, value in (
        ("the teacher's weights digest", teacher.get("weights_sha256")),
        ("the roots digest", record.get("roots_sha256")),
    ):
        try:
            benchmark_core.require_sha(value)
        except ValueError as error:
            raise PackError(f"{key} is unusable") from error
    return record


def build_manifest(
    directory: Path, record: dict, *, version: str, kind: str, license_: str
) -> dict:
    """The manifest the contract defines, from the record the generator left."""
    _require(kind in KINDS, "unverified dataset origin")
    _require(license_.strip() not in ("", "unknown", "unresolved"), "missing data terms")
    _require((directory / GROUPS_FILE).is_file(), "the corpus has no groups")
    _require(
        (directory / LICENSE_FILE).is_file(),
        "the corpus has no data terms beside it; pass the evidence file",
    )

    groups = json.loads((directory / GROUPS_FILE).read_text(encoding="utf-8"))
    _require(isinstance(groups, list) and groups, "the corpus has no groups")
    # The manifest has to state the counts, and `load_groups` will not read a manifest that does
    # not — so the counts are taken from the payload before anything has validated it. That makes
    # the shape this module's problem: without these two lines a malformed group leaves as a
    # `KeyError` past every caller that knows how to report a refusal.
    _require(
        all(isinstance(group, dict) for group in groups),
        "the corpus holds something that is not a group",
    )
    _require(
        all(isinstance(group.get("candidates"), list) for group in groups),
        "a group does not carry a list of candidates",
    )

    # The columns and the golden digest come from the committed corpus for the engine the
    # generator says it ran, never from the generation record: a producer does not get to state
    # the layout it was supposed to produce.
    columns, golden_digest = golden_columns(
        str(record["feature_schema"]), str(record["engine_version"])
    )
    declared = tuple(record.get("columns", ()))
    _require(
        declared == () or declared == columns,
        "the generator's columns are not the engine's layout for that schema",
    )

    return {
        "schema": SCHEMA_ID,
        "version": version,
        "perspective": PERSPECTIVE,
        "target": TARGET,
        "kind": kind,
        "license": license_,
        "feature_schema": record["feature_schema"],
        "engine_version": record["engine_version"],
        "columns": list(columns),
        "teacher": {
            "id": record["teacher"]["id"],
            "weights_sha256": record["teacher"]["weights_sha256"],
        },
        "groups": len(groups),
        "candidates": sum(len(group["candidates"]) for group in groups),
        "groups_sha256": kcp13.sha256_of(directory / GROUPS_FILE),
        "license_evidence_sha256": kcp13.sha256_of(directory / LICENSE_FILE),
        "source_sha256": record["roots_sha256"],
        "golden_sha256": golden_digest,
    }


def pack(
    directory: Path,
    *,
    license_: str,
    license_file: Path | None = None,
    version: str = DEFAULT_VERSION,
    kind: str = DEFAULT_KIND,
) -> tuple[dict, dict]:
    """Write the manifest, then admit the corpus through the loader that will admit it later.

    Returns the manifest and a summary of what was admitted. The summary is the part an operator
    reads: a corpus is worth what its lists are, and the counts that decide that — how many groups
    are larger than the shortlist, how the splits fell — are not visible in a 200 MB file.
    """
    directory = Path(directory).resolve()
    record = read_generation(directory)
    if license_file is not None:
        _require(Path(license_file).is_file(), "the data terms file does not exist")
        shutil.copyfile(license_file, directory / LICENSE_FILE)

    manifest = build_manifest(directory, record, version=version, kind=kind, license_=license_)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    admitted, groups = _admit(directory, payload)
    (directory / MANIFEST_FILE).write_text(payload, encoding="utf-8")
    return admitted, summarise(groups, record)


def _admit(directory: Path, payload: str) -> tuple[dict, list[dict]]:
    """Try a manifest the corpus does not have yet, and only then let it have it.

    `load_groups` reads the manifest from a fixed name beside the data, so the only way to try one
    without committing to it is to try it somewhere else. The payload is hard-linked into a
    staging directory on the same filesystem — a corpus is hundreds of megabytes, and a link is
    the same bytes, the same inode and therefore the same digest — which is the shape
    `dicechess_training.hub` already uses to admit an artifact before publishing it. A copy is the
    fallback, because the link is an optimisation and not part of the guarantee.

    The guarantee is what this buys: a refused corpus is left exactly as it was found. No manifest
    appears beside it, and one that was already there is untouched — its corpus was admissible
    until this command ran, and a failed run must not be what changes that. Writing first and
    undoing afterwards would give the same outcome only for as long as the undo kept working.
    """
    staging = Path(tempfile.mkdtemp(dir=directory, prefix=".prerank-staging-"))
    try:
        for name in (GROUPS_FILE, LICENSE_FILE):
            try:
                os.link(directory / name, staging / name)
            except OSError:
                shutil.copy2(directory / name, staging / name)
        (staging / MANIFEST_FILE).write_text(payload, encoding="utf-8")
        return load_groups(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def summarise(groups: list[dict], record: dict, shortlist: int = 48) -> dict:
    """What the corpus is, in the terms that decide whether it can answer a ranking question.

    `shortlist` is the production `candidateLimit`. Groups no larger than it are decisions a
    pre-ranker cannot get wrong — every candidate survives — so a rank metric has to name the
    subset it was measured on, and that subset is worth knowing before training rather than after.
    """
    sizes = sorted(len(group["candidates"]) for group in groups)
    splits = Counter(assign_splits(groups))
    distinct_targets = sorted(len({c["target"] for c in g["candidates"]}) for g in groups)
    return {
        "groups": len(groups),
        "candidates": sum(sizes),
        "splits": {name: splits.get(name, 0) for name in ("train", "validation", "test")},
        "candidates_per_group": {
            "min": sizes[0],
            "median": sizes[len(sizes) // 2],
            "p95": sizes[int(len(sizes) * 0.95)],
            "max": sizes[-1],
        },
        "groups_above_shortlist": sum(1 for size in sizes if size > shortlist),
        "groups_entirely_tied": sum(1 for count in distinct_targets if count == 1),
        "median_distinct_targets": distinct_targets[len(distinct_targets) // 2],
        "roots_skipped": record.get("skipped", {}),
        "collapsed_transpositions": record.get("collapsed_transpositions"),
    }
