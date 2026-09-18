"""Issue the preregistered seal that unlocks final qualification.

Ten fields bind the implementation, the protocol, both datasets, the candidate, the accepted
reference inventory, the promotion comparator, the frozen concurrency workload and the serving
envelope. Nine are derivable from artifacts that already exist, and deriving them is the point:
copying five SHA-256 values between files by hand is where a transcription error lands, and the
failure mode is the expensive one, because the seal is the last lock before a single-use holdout
is opened.

So everything this can refuse, it refuses here rather than at qualification. A final bundle that
overlaps development, a reference that was not trained on the same data, a promotion comparator
the rules do not allow — each of those is accepted by a hand-written seal and rejected only when
the final command runs, which is the worst possible ordering for a holdout that can be opened once.

What it cannot derive it does not invent: the serving limits and the concurrency workload digest
come from the owner, who froze them before any result existed.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.benchmark.splits import position_key
from dicechess_training.contracts import kcp13
from dicechess_training.publish import stage, write_once

SEAL_SCHEMA = "playground-seal-v1"
NO_INFORMATION = "no-information"


class SealError(ValueError):
    """A seal could not be issued. Messages never carry paths, identities or limits."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealError(message)


def _is_number(value: Any) -> bool:
    """A real number, not a boolean wearing one: `isinstance(True, int)` is true in Python."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _admit(directory, data_manifest, protocol):
    """Admit a package exactly as the benchmark admits it, so the seal cannot bind a worse rule."""
    try:
        return benchmark_core.load_candidate(directory, data_manifest, protocol)
    except ValueError as error:
        raise SealError("a package is not one the benchmark would admit") from error


def _check_final_is_qualifiable(final_rows, development_rows) -> dict[str, int]:
    """Refuse a bundle that could never qualify, before it consumes the seal.

    `prepare_final` applies exactly these two rules. Applying them here costs nothing and turns a
    spent holdout into a message.
    """
    try:
        benchmark_core.check_sealed_disjointness(final_rows, development_rows)
    except ValueError as error:
        raise SealError("the final bundle shares a group, game or root with development") from error
    seen = {position_key(row) for row in development_rows}
    overlap = len(seen & {position_key(row) for row in benchmark_core.decisive(final_rows)})
    _require(not overlap, "the final bundle repeats a position development has already seen")
    return {"final_rows": len(final_rows), "development_rows": len(development_rows)}


def build_seal(
    candidate_dir: str | Path,
    development_data_dir: str | Path,
    final_data_dir: str | Path,
    *,
    concurrency_workload_sha256: str,
    latency_p95_ms: float,
    rss_mb: float,
    reference_dirs=(),
    promotion_reference: str | None = None,
) -> dict[str, Any]:
    """Derive the seal, or raise. Writes nothing; publication is the caller's."""
    protocol = benchmark_core.load_protocol()

    development, development_rows = benchmark_core.load_dataset(development_data_dir)
    _require(development["kind"] != "synthetic", "synthetic development cannot qualify a model")
    final, final_rows = benchmark_core.load_dataset(final_data_dir)
    _require(final["kind"] == "owner-controlled", "the final bundle must be owner-controlled")
    _check_final_is_qualifiable(final_rows, development_rows)

    candidate = _admit(candidate_dir, development, protocol)
    try:
        benchmark_core.check_training_identity(candidate, development, development_rows)
    except ValueError as error:
        raise SealError("the candidate was not trained on this development bundle") from error

    # Every reference is admitted and must share the candidate's training identity, because the
    # comparison the gate makes is only meaningful between models fitted on the same data.
    accepted = []
    for directory in reference_dirs:
        manifest = _admit(directory, development, protocol)
        try:
            benchmark_core.check_training_identity(manifest, development, development_rows)
        except ValueError as error:
            raise SealError("a reference was not trained on this development bundle") from error
        accepted.append(benchmark_core.digest(manifest))
    _require(len(accepted) == len(set(accepted)), "the same reference was supplied twice")

    # `reference_predictions` allows the no-information comparator only while no model has been
    # accepted; once one has, comparing against a constant would understate what is already served.
    if accepted:
        _require(
            promotion_reference is not None and promotion_reference != NO_INFORMATION,
            "an accepted reference exists, so the comparator cannot be the no-information one",
        )
        _require(promotion_reference in accepted, "the comparator is not an accepted reference")
    else:
        _require(
            promotion_reference in (None, NO_INFORMATION),
            "there is no accepted reference for the comparator to name",
        )
        promotion_reference = NO_INFORMATION

    try:
        benchmark_core.require_sha(concurrency_workload_sha256)
    except ValueError as error:
        raise SealError("the concurrency workload digest is unusable") from error
    limits = {"latency_p95_ms": latency_p95_ms, "rss_mb": rss_mb}
    for value in limits.values():
        _require(
            _is_number(value) and np.isfinite(value) and value > 0,
            "a serving limit is missing or is not a positive number",
        )

    return {
        "schema": SEAL_SCHEMA,
        "implementation_sha256": benchmark_core.implementation_digest(),
        "benchmark_sha256": benchmark_core.digest(protocol),
        "development_dataset_sha256": benchmark_core.digest(development),
        "final_dataset_sha256": benchmark_core.digest(final),
        "candidate_manifest_sha256": benchmark_core.digest(candidate),
        "accepted_references": accepted,
        "promotion_reference": promotion_reference,
        "concurrency_workload_sha256": concurrency_workload_sha256,
        "serving_limits": {key: float(value) for key, value in limits.items()},
    }


def publish_seal(seal: dict[str, Any], path: str | Path) -> str:
    """Write the seal and return its own SHA-256, which the owner retains independently.

    Published write-once: a preregistration that a later run can overwrite is not one, and the
    check-then-rename this used to do is a race two issuers can both pass.
    """
    path = Path(path)
    _require(not path.exists(), "a seal already exists at that location")
    with contextlib.ExitStack() as stack:
        staged = stage(
            path, json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n", stack
        )
        digest = kcp13.sha256_of(staged)
        try:
            write_once([(staged, path)])
        except FileExistsError as error:
            raise SealError("a seal already exists at that location") from error
        return digest
