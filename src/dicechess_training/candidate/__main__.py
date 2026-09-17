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
    # The model identity stays in the package manifest: the card template counts model
    # identities among the fields that do not belong in a shareable summary.
    return {
        "schema": "playground-candidate-v1",
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
    reserved = None
    try:
        args = parser.parse_args(argv)
        # The report destination is claimed before the build: discovering that it is taken
        # afterwards would leave a package behind that this command reported as failed.
        report_path = Path(args.report) if args.report else None
        if report_path is not None:
            report_path.open("x", encoding="utf-8").close()
            reserved = report_path
        # The ONNX exporter writes progress to stdout; this entry point promises JSON there,
        # so library chatter is redirected to stderr for the duration of the build.
        with contextlib.redirect_stdout(sys.stderr):
            summary = build_candidate(
                args.data, args.output, CandidateConfig(seed=args.seed, model_id=args.model_id)
            )
        result = (
            json.dumps(_public_summary(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if report_path is not None:
            report_path.write_text(result, encoding="utf-8")
        else:
            print(result, end="")
        return 0
    except Exception:
        # Includes runtime/IO errors: third-party exception text may carry private paths.
        if reserved is not None:
            reserved.unlink(missing_ok=True)
        print(json.dumps({"schema": "playground-error-v1", "error": "candidate-build-failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
