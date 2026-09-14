"""CLI entrypoint for running the feature schema ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dicechess_training.ablation.report import render_markdown_report
from dicechess_training.ablation.runner import run_ablation

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predeclared feature schema ablation.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "docs/ablation/protocol-v1.json",
        help="Path to protocol JSON file.",
    )
    parser.add_argument(
        "--enriched-dir",
        type=Path,
        default=ROOT / "data/enriched",
        help="Base directory containing enriched schema shard directories.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "docs/ablation/ablation-report.json",
        help="Path to write output JSON report.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "docs/ablation/report.md",
        help="Path to write output Markdown report.",
    )

    args = parser.parse_args()

    report = run_ablation(args.protocol, args.enriched_dir)

    # Save JSON report
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote JSON report to: {args.output_json}")

    # Generate and save Markdown report
    md_content = render_markdown_report(report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write(md_content)
    print(f"Wrote Markdown report to: {args.output_md}")

    # Print summary
    dec_schema = report["decision"]["selected_schema"]
    dec_id = report["decision"]["selected_schema_id"]
    print(f"DECISION: Selected schema {dec_schema} ({dec_id})")
    print(f"Gate Results: {report['decision']['gate_results']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
