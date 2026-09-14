"""CLI entrypoint for running the feature schema ablation."""

from __future__ import annotations

import json

from dicechess_training.ablation.report import render_markdown_report
from dicechess_training.ablation.runner import (
    DEFAULT_PROTOCOL_PATH,
    ROOT,
    run_ablation,
)

OUTPUT_JSON_PATH = ROOT / "docs/ablation/ablation-report.json"
OUTPUT_MD_PATH = ROOT / "docs/ablation/report.md"


def main() -> None:
    report = run_ablation(protocol_path=DEFAULT_PROTOCOL_PATH)

    # Save JSON report
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote JSON report to: {OUTPUT_JSON_PATH}")

    # Generate and save Markdown report
    md_content = render_markdown_report(report)
    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Wrote Markdown report to: {OUTPUT_MD_PATH}")

    # Print summary
    dec_schema = report["decision"]["selected_schema"]
    dec_id = report["decision"]["selected_schema_id"]
    print(f"DECISION: Selected schema {dec_schema} ({dec_id})")
    print(f"Gate Results: {report['decision']['gate_results']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
