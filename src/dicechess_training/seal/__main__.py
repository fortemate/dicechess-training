"""JSON-only entry point. Failures never print paths, identities or limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .issue import build_seal, publish_seal


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid CLI arguments")


def _input_directory(raw: str) -> Path:
    resolved = Path(raw).resolve()
    if not resolved.is_dir():
        raise ValueError("input directory does not exist")
    return resolved


def _output_location(raw: str) -> Path:
    path = Path(raw)
    if path.is_symlink():
        raise ValueError("output path must not be a symlink")
    return path.resolve()


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--development-data", required=True)
    parser.add_argument("--final-data", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--promotion-reference")
    parser.add_argument("--concurrency-workload-sha256", required=True)
    parser.add_argument("--latency-p95-ms", type=float, required=True)
    parser.add_argument("--rss-mb", type=float, required=True)
    parser.add_argument("--output", required=True)
    try:
        args = parser.parse_args(argv)
        seal = build_seal(
            _input_directory(args.candidate),
            _input_directory(args.development_data),
            _input_directory(args.final_data),
            concurrency_workload_sha256=args.concurrency_workload_sha256,
            latency_p95_ms=args.latency_p95_ms,
            rss_mb=args.rss_mb,
            reference_dirs=[_input_directory(d) for d in args.reference],
            promotion_reference=args.promotion_reference,
        )
        digest = publish_seal(seal, _output_location(args.output))
        # The digest is the whole point of printing anything: qualification takes the value the
        # owner retained, never one recomputed from the file at the time it is checked.
        print(
            json.dumps(
                {
                    "schema": seal["schema"],
                    "seal_sha256": digest,
                    "accepted_references": len(seal["accepted_references"]),
                    "promotion_reference": seal["promotion_reference"],
                    "retain": "Store seal_sha256 in the append-only record now; "
                    "qualification takes the retained value, not a recomputed one.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception:
        print(json.dumps({"schema": "playground-error-v1", "error": "seal-issue-failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
