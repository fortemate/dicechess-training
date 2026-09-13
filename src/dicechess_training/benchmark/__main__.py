"""JSON-only entry point. Failures never print input paths or artifact metadata."""

import argparse
import json
from pathlib import Path

from .core import evaluate


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid CLI arguments")


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--mode", choices=("development", "final"), default="development")
    parser.add_argument("--development-data")
    parser.add_argument("--seal")
    parser.add_argument("--seal-sha256")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--serving-evidence")
    parser.add_argument("--output")
    try:
        args = parser.parse_args(argv)
        report = evaluate(
            args.data,
            args.candidate,
            mode=args.mode,
            development_dir=args.development_data,
            seal_path=args.seal,
            seal_sha256=args.seal_sha256,
            reference_dirs=args.reference,
            evidence_path=args.serving_evidence,
        )
        result = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as output:
                output.write(result)
        else:
            print(result, end="")
        return (
            0
            if args.mode == "development"
            or report["decision"]["status"] == "eligible-for-owner-review"
            else 1
        )
    except Exception:
        # Includes runtime/IO errors: third-party exception text may contain private paths.
        print(json.dumps({"schema": "playground-error-v1", "error": "invalid-benchmark-input"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
