"""CLI entrypoint for running the feature schema ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dicechess_training.ablation.report import render_markdown_report
from dicechess_training.ablation.runner import (
    DEFAULT_PROTOCOL_PATH,
    ROOT,
    run_ablation,
)

DEFAULT_OUTPUT_DIR = ROOT / "out/ablation"


def _write_exclusive(path: Path, content: str, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination file already exists: {path}. "
            "Use --overwrite to replace existing evidence."
        )
    path.write_text(content, encoding="utf-8")


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run feature schema ablation across candidate schemas."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Path to the protocol JSON definition",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/enriched",
        help="Base directory containing enriched schema shards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output reports (default: out/ablation)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Explicit output path for JSON report (overrides --output-dir)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Explicit output path for Markdown report (overrides --output-dir)",
    )
    parser.add_argument(
        "--extraction-cost",
        type=Path,
        default=ROOT / "tests/fixtures/benchmark/extraction-cost-0.9.3.json",
        help="Path to verified JVM extraction benchmark JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow replacing existing output report files",
    )

    parsed = parser.parse_args(args)

    report = run_ablation(
        protocol_path=parsed.protocol,
        enriched_base_dir=parsed.data_dir,
        extraction_cost_path=parsed.extraction_cost,
    )

    out_json = parsed.output_json or (parsed.output_dir / "ablation-report.json")
    out_md = parsed.output_md or (parsed.output_dir / "report.md")

    # Save JSON report exclusively
    _write_exclusive(out_json, json.dumps(report, indent=2), overwrite=parsed.overwrite)
    print(f"\nWrote JSON report to: {out_json}")

    # Generate and save Markdown report exclusively
    md_content = render_markdown_report(report)
    _write_exclusive(out_md, md_content, overwrite=parsed.overwrite)
    print(f"Wrote Markdown report to: {out_md}")

    # Print summary
    dec_schema = report["decision"]["selected_schema"]
    dec_id = report["decision"]["selected_schema_id"]
    print(f"DECISION: Selected schema {dec_schema} ({dec_id})")
    print(f"Gate Results: {report['decision']['gate_results']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
