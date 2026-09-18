"""JSON-only entry point. Failures never print paths, hosts, service details or measurements."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .evidence import build_evidence


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid CLI arguments")


def _input_directory(raw: str) -> Path:
    resolved = Path(raw).resolve()
    if not resolved.is_dir():
        raise ValueError("input directory does not exist")
    return resolved


def _input_file(raw: str) -> Path:
    resolved = Path(raw).resolve()
    if not resolved.is_file():
        raise ValueError("input file does not exist")
    return resolved


def _output_location(raw: str) -> Path:
    path = Path(raw)
    if path.is_symlink():
        raise ValueError("output path must not be a symlink")
    return path.resolve()


def _public_summary(result: dict) -> dict:
    """Digests and failed check names only.

    The document's own rule: reports emit an evidence digest and failed check names, never limits,
    latencies or raw service detail. The measurements stay in the file the owner retains.
    """
    evidence = result["evidence"]
    return {
        "schema": evidence["schema"],
        "candidate_manifest_sha256": evidence["candidate_manifest_sha256"],
        "probe_suite_sha256": evidence["probe_suite_sha256"],
        "raw_evidence_sha256": evidence["raw_evidence_sha256"],
        "checks_passed": sum(1 for passed in evidence["checks"].values() if passed),
        "checks_total": len(evidence["checks"]),
        "failed_checks": result["failed_checks"],
    }


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--training-data", required=True, help="The bundle the candidate was trained on"
    )
    parser.add_argument(
        "--qualification-data",
        action="append",
        required=True,
        help="A dataset whose every row is covered by parity and bounds; repeatable",
    )
    parser.add_argument("--observations", required=True, help="Owner-reviewed service observations")
    parser.add_argument("--output", required=True, help="Serving evidence document to write")
    parser.add_argument("--raw-output", required=True, help="Raw observations to retain privately")
    try:
        args = parser.parse_args(argv)
        # Recovering the model runs the packager's own training and ONNX export, which write
        # progress to stdout; this entry point promises JSON there, so their chatter goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = build_evidence(
                _input_directory(args.candidate),
                _input_directory(args.training_data),
                [_input_directory(d) for d in args.qualification_data],
                _input_file(args.observations),
                _output_location(args.output),
                _output_location(args.raw_output),
            )
        print(json.dumps(_public_summary(result), indent=2, sort_keys=True, allow_nan=False))
        return 0 if not result["failed_checks"] else 1
    except Exception:
        print(json.dumps({"schema": "playground-error-v1", "error": "serving-evidence-failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
