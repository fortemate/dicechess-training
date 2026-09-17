"""JSON-only entry point. Failures never print input paths or artifact metadata."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .build import CandidateConfig, build_candidate


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid CLI arguments")


def _public_summary(summary: dict) -> dict:
    """Digests and counts only: no dataset path, row content or output location."""
    return {
        "schema": "playground-candidate-v1",
        "model_id": summary["model_id"],
        "model_sha256": summary["manifest"]["modelSha256"],
        "engine_compatibility": summary["manifest"]["engineCompatibility"],
        "provenance": summary["provenance"],
        "parity": summary["parity"],
        "training_rows": summary["training_rows"],
        "inner_tuning_rows": summary["inner_tuning_rows"],
        "selected_epochs": summary["epoch_selection"]["selected_epochs"],
    }


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--report")
    try:
        args = parser.parse_args(argv)
        # The ONNX exporter writes progress to stdout; this entry point promises JSON there,
        # so library chatter is redirected to stderr for the duration of the build.
        with contextlib.redirect_stdout(sys.stderr):
            summary = build_candidate(
                args.data, args.output, CandidateConfig(seed=args.seed, model_id=args.model_id)
            )
        result = (
            json.dumps(_public_summary(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if args.report:
            with Path(args.report).open("x", encoding="utf-8") as report:
                report.write(result)
        else:
            print(result, end="")
        return 0
    except Exception:
        # Includes runtime/IO errors: third-party exception text may carry private paths.
        print(json.dumps({"schema": "playground-error-v1", "error": "candidate-build-failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
