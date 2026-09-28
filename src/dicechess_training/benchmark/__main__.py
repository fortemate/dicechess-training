"""JSON-only entry point. Failures never print input paths or artifact metadata."""

import argparse
import json
from pathlib import Path

from .core import evaluate


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid CLI arguments")


def _workspace_path(raw: str | None) -> Path | None:
    """CLI file arguments stay under the invocation directory, including symlinks."""
    if raw is None:
        return None
    resolved = Path(raw).resolve()
    if not resolved.is_relative_to(Path.cwd().resolve()):
        raise ValueError("path is outside the working directory")
    return resolved


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
        output_path = _workspace_path(args.output)
        report = evaluate(
            _workspace_path(args.data),
            _workspace_path(args.candidate),
            mode=args.mode,
            development_dir=_workspace_path(args.development_data),
            seal_path=_workspace_path(args.seal),
            seal_sha256=args.seal_sha256,
            reference_dirs=[_workspace_path(path) for path in args.reference],
            evidence_path=_workspace_path(args.serving_evidence),
        )
        result = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if output_path is not None:
            with output_path.open("x", encoding="utf-8") as output:
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
