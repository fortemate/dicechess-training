"""CLI entrypoint for running the feature schema ablation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from dicechess_training.ablation.paths import resolve_read_path
from dicechess_training.ablation.report import render_markdown_report
from dicechess_training.ablation.runner import (
    DEFAULT_PROTOCOL_PATH,
    ROOT,
    run_ablation,
)

DEFAULT_OUTPUT_DIR = ROOT / "out/ablation"


def _safe_write_path(path: Path) -> Path:
    """Validate and resolve target path to prevent path traversal."""
    resolved = path.resolve()
    base_repo = ROOT.resolve()
    base_tmp = Path(tempfile.gettempdir()).resolve()
    base_cwd = Path.cwd().resolve()
    if not (
        resolved.is_relative_to(base_repo)
        or resolved.is_relative_to(base_tmp)
        or resolved.is_relative_to(base_cwd)
    ):
        raise ValueError(f"Path traversal detected: {resolved} is outside allowed boundaries")
    return resolved


def _write_exclusive(path: Path, content: str, overwrite: bool = False) -> None:
    safe_path = _safe_write_path(path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    try:
        with safe_path.open(mode, encoding="utf-8") as f:
            f.write(content)
    except FileExistsError:
        raise FileExistsError(
            f"Destination file already exists: {safe_path}. "
            "Use --overwrite to replace existing evidence."
        ) from None


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

    protocol_path = resolve_read_path(parsed.protocol)
    data_dir = resolve_read_path(parsed.data_dir, directory=True)
    ext_cost_path = resolve_read_path(parsed.extraction_cost) if parsed.extraction_cost else None

    report = run_ablation(
        protocol_path=protocol_path,
        enriched_base_dir=data_dir,
        extraction_cost_path=ext_cost_path,
    )

    target_json = parsed.output_json or (parsed.output_dir / "ablation-report.json")
    target_md = parsed.output_md or (parsed.output_dir / "report.md")
    out_json = _safe_write_path(target_json)
    out_md = _safe_write_path(target_md)

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
    print(f"Gate Results: {report['decision']['gate_results']}")
    if dec_schema is None:
        print(f"DECISION: none — {report['decision']['inadmissible_reason']}")
        print("=" * 60 + "\n")
        # The reports above are written first: the evidence of a failed run is kept, and the
        # command still fails so no pipeline reads "no selection" as a silent pass.
        raise SystemExit(1)
    print(f"DECISION: Selected schema {dec_schema} ({dec_id})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
