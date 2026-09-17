"""JSON-only entry point for dataset export. Failures never print private paths."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .export import DatasetConfig, build_dataset


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid CLI arguments")


def _input_location(raw: str) -> Path:
    resolved = Path(raw).resolve()
    if not resolved.exists():
        raise ValueError("input path does not exist")
    return resolved


def _output_location(raw: str) -> Path:
    path = Path(raw)
    if path.is_symlink():
        raise ValueError("output path must not be a symlink")
    return path.resolve()


def _public_summary(summary: dict) -> dict:
    """Sanitized summary: counts, schema, and digests only."""
    return {
        "schema": summary["schema"],
        "version": summary["version"],
        "kind": summary["kind"],
        "engine_version": summary["engine_version"],
        "rows": summary["rows"],
        "games": summary["games"],
        "manifest_sha256": summary["manifest_sha256"],
        "rows_sha256": summary["rows_sha256"],
        "golden_sha256": summary["golden_sha256"],
        "source_sha256": summary["source_sha256"],
    }


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument(
        "--shards", required=True, help="Input directory of enriched Parquet shards"
    )
    parser.add_argument("--output", required=True, help="Target output directory")
    parser.add_argument(
        "--kind",
        choices=["owner-controlled", "public-licensed", "synthetic"],
        default="owner-controlled",
        help="Dataset origin kind",
    )
    parser.add_argument("--engine-version", help="Engine version override")
    parser.add_argument(
        "--license",
        default="Fortemate Owner-Controlled Data Policy v1",
        dest="license_name",
    )
    parser.add_argument("--license-file", help="Path to custom license text")
    parser.add_argument("--max-games", type=int, help="Optional game limit")
    parser.add_argument("--version", default="1.0.0", help="Semantic version")
    parser.add_argument("--source-sha256", help="Precomputed source SHA-256")
    parser.add_argument("--report", help="Path to write JSON summary")

    reserved = None
    try:
        args = parser.parse_args(argv)
        shards_path = _input_location(args.shards)
        output_dir = _output_location(args.output)
        report_path = _output_location(args.report) if args.report else None
        if report_path is not None:
            report_path.open("x", encoding="utf-8").close()
            reserved = report_path

        config = DatasetConfig(
            kind=args.kind,
            version=args.version,
            license_name=args.license_name,
            license_file=Path(args.license_file).resolve() if args.license_file else None,
            engine_version=args.engine_version,
            max_games=args.max_games,
            source_sha256=args.source_sha256,
        )

        with contextlib.redirect_stdout(sys.stderr):
            summary = build_dataset(shards_path, output_dir, config)

        result = (
            json.dumps(_public_summary(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if report_path is not None:
            report_path.write_text(result, encoding="utf-8")
        else:
            print(result, end="")
        return 0
    except Exception:
        if reserved is not None:
            reserved.unlink(missing_ok=True)
        err = json.dumps(
            {
                "error": "dataset build failed",
                "details": "invalid dataset specification or build failure",
            }
        )
        print(err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
