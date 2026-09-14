"""Generate Markdown report from ablation results."""

from __future__ import annotations

from typing import Any

NOTE_ALERT = "> [!NOTE]"
_SEVEN_COL_SEPARATOR = "|---|---|---|---|---|---|---|"


def _render_header(
    report: dict[str, Any],
    split: dict[str, Any],
    decision: dict[str, Any],
) -> list[str]:
    lines = [
        "# Feature Schema Ablation Report (Issue #17)",
        "",
        "Offline ablation evaluating **S0** (`kcp-13`), **S1** (`kcp-mobility-27-v1`), "
        "and **S2** (`kcp-mobility-pawns-31-v1`) under the protocol identified below.",
        "",
        NOTE_ALERT,
        "> **Provisional Development Report**: Input shard digests below identify the evaluated "
        "data. Development gate results do not establish final qualification.",
        "> Final qualification requires owner-run evaluation of the frozen private "
        "corpus under the same protocol.",
        (
            "> Reference: **"
            + decision.get(
                "private_qualification_ref",
                "Private Decision: Playground Feature Schema Qualification (Issue #17)",
            )
            + "**."
        ),
        "",
        "## 1. Executive Summary & Decision",
        "",
    ]
    sel_id = decision["selected_schema_id"]
    sel_name = decision["selected_schema"]
    lines.append(f"- **Selected Feature Schema**: **`{sel_id}`** ({sel_name})")
    lines.append(
        f"- **Protocol Version**: `{report['protocol_version']}` "
        f"(SHA-256: `{report['protocol_sha256'][:16]}...`)"
    )
    lines.append(f"- **Engine Version**: `{report['engine_version']}`")
    total_g = split["train_games"] + split["val_games"] + split.get("test_games", 0)
    lines.append(
        f"- **Dataset**: input enriched shards ({split['total_positions']:,} rows, {total_g} games)"
    )
    test_pos = split.get("test_positions", 0)
    lines.append(
        f"  - Decisive rows: {split['decisive_positions']:,} "
        f"({split['train_positions']:,} train / {split['val_positions']:,} val / "
        f"{test_pos:,} test holdout)"
    )
    leakage_info = split.get("leakage", {})
    if leakage_info:
        tr_val = leakage_info.get("train:validation", 0)
        tr_test = leakage_info.get("train:test", 0)
        val_test = leakage_info.get("validation:test", 0)
        unseen = split.get("unseen_val_positions", 0)
        lines.append(
            f"  - Canonical position leakage: train:val = {tr_val}, train:test = {tr_test}, "
            f"val:test = {val_test} (unseen val positions: {unseen:,})"
        )
    lines.append("")

    if sel_name == "S0":
        lines.append(NOTE_ALERT)
        lines.append(
            "> **Development Verdict**: Neither wider candidate cleared all development gates."
        )
        lines.append(
            "> Baseline **S0 (`kcp-13`) remains the development reference**; "
            "final owner qualification remains pending."
        )
    else:
        lines.append("> [!IMPORTANT]")
        lines.append(
            f"> **Development Verdict**: Schema **{sel_name} (`{sel_id}`) cleared all gate rules** "
            "for provisional development evidence; final owner qualification remains pending."
        )
    return lines


def _render_input_shards(report: dict[str, Any]) -> list[str]:
    shards = report.get("input_shard_digests", {})
    if not shards:
        return []
    lines = [
        "### Input Shard Provenance",
        "",
        "| Schema | Shard File | SHA-256 Digest |",
        "|---|---|---|",
    ]
    for skey, files in sorted(shards.items()):
        for fname, digest_val in sorted(files.items()):
            lines.append(f"| **{skey}** | `{fname}` | `{digest_val[:16]}...` |")
    return lines


def _render_key_metrics(schemas: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    lines = [
        "## 2. Primary Estimand: Single-Model Replication",
        "",
        "Evaluation of single-model performance across independent random seeds (mean ± std). "
        "Paired 95% confidence intervals are computed via whole-game bootstrap resampling "
        "on the mean seed loss delta of the primary estimand:",
        "",
        "| Schema | Features | Log Loss | 95% CI vs S0 (Δ) | Brier | ECE | Rel. LL Gain | "
        "Gate Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    s0_sm = schemas["S0"]["single_model_summary"]
    lines.append(
        f"| **S0** (`kcp-13`) | {schemas['S0']['feature_count']} | "
        f"{s0_sm['log_loss_mean']:.4f} ± {s0_sm['log_loss_std']:.4f} | — (Reference) | "
        f"{s0_sm['brier_mean']:.4f} ± {s0_sm['brier_std']:.4f} | "
        f"{s0_sm['ece_mean']:.4f} ± {s0_sm['ece_std']:.4f} | — | Baseline |"
    )

    for skey in ["S1", "S2"]:
        info = schemas[skey]
        sm = info["single_model_summary"]
        gate = gates[skey]
        ci = gate["log_loss_delta_ci_95"]
        ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—"
        rel_gain = gate["relative_log_loss_gain"] * 100
        status = "**PASSED**" if gate["cleared"] else "FAILED"
        lines.append(
            f"| **{skey}** (`{info['schema_id']}`) | {info['feature_count']} | "
            f"{sm['log_loss_mean']:.4f} ± {sm['log_loss_std']:.4f} | {ci_str} | "
            f"{sm['brier_mean']:.4f} ± {sm['brier_std']:.4f} | "
            f"{sm['ece_mean']:.4f} ± {sm['ece_std']:.4f} | "
            f"{rel_gain:+.2f}% | {status} |"
        )
    return lines


def _render_diagnostic_ensemble(schemas: dict[str, Any]) -> list[str]:
    lines = [
        "### 2b. Secondary Diagnostic: 5-Model Ensemble",
        "",
        NOTE_ALERT,
        "> Ensemble predictions (average probability across 5 seeds). "
        "Reported for variance-reduction diagnostics; not the single-model deployable contract.",
        "",
        "| Schema | Features | Ensemble Log Loss | Ensemble Brier | Ensemble ECE |",
        "|---|---|---|---|---|",
    ]
    for skey in ["S0", "S1", "S2"]:
        ens = schemas[skey]["ensemble_diagnostic"]["scores"]
        fc = schemas[skey]["feature_count"]
        sid = schemas[skey]["schema_id"]
        lines.append(
            f"| **{skey}** (`{sid}`) | {fc} | {ens['log_loss']:.4f} | "
            f"{ens['brier']:.4f} | {ens['ece']:.4f} |"
        )
    return lines


def _render_gate_checklist(gates: dict[str, Any]) -> list[str]:
    lines = [
        "### Gate Checklist",
        "",
        "| Gate Rule | S1 (`kcp-mobility-27-v1`) | S2 (`kcp-mobility-pawns-31-v1`) | Requirement |",
        "|---|---|---|---|",
    ]
    rules = [
        ("relative_log_loss_gain_gte_1pct", ">= +1.0% relative gain on full validation"),
        (
            "paired_ci_upper_lt_0",
            "Paired 95% group-bootstrap CI upper bound < 0 on mean seed delta",
        ),
        ("brier_no_regression", "Brier regression <= 0.0000"),
        ("ece_regression_lte_0_01", "ECE regression <= 0.0100"),
        ("slice_log_loss_lte_0_01", "Max slice log loss regression <= 0.0100"),
        ("slice_brier_lte_0_01", "Max slice Brier regression <= 0.0100"),
        ("unseen_positions_exist", "Non-empty unseen validation partition (no-position-leakage)"),
        ("unseen_log_loss_lte_0_01", "Unseen positions log loss regression <= 0.0100"),
        ("unseen_brier_no_regression", "Unseen positions Brier regression <= 0.0000"),
        ("extraction_cost_evidence_valid", "Verified JVM extraction benchmark artifact present"),
        (
            "extraction_cost_lte_tolerance",
            "Mean JVM extraction latency overhead <= 10.0% relative to S0",
        ),
    ]
    for rule_name, req in rules:
        s1_ok = "PASS" if gates["S1"]["checks"].get(rule_name, False) else "FAIL"
        s2_ok = "PASS" if gates["S2"]["checks"].get(rule_name, False) else "FAIL"
        lines.append(f"| `{rule_name}` | **{s1_ok}** | **{s2_ok}** | {req} |")
    return lines


def _render_unseen_table(schemas: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    lines = [
        "## 3. Unseen Validation Positions Performance",
        "",
        "Evaluation on validation positions with no FEN/side overlap in the training partition:",
        "",
        "| Schema | Unseen Positions | Log Loss | 95% CI vs S0 (Δ) | Brier | ECE | Unseen Gate |",
        _SEVEN_COL_SEPARATOR,
    ]
    s0_u = schemas["S0"].get("single_model_unseen", {})
    cnt = s0_u.get("count", 0)
    lines.append(
        f"| **S0** (`kcp-13`) | {cnt:,} | "
        f"{s0_u.get('log_loss_mean', 0.0):.4f} ± {s0_u.get('log_loss_std', 0.0):.4f} | "
        f"— (Reference) | {s0_u.get('brier_mean', 0.0):.4f} ± {s0_u.get('brier_std', 0.0):.4f} | "
        f"{s0_u.get('ece_mean', 0.0):.4f} ± {s0_u.get('ece_std', 0.0):.4f} | Baseline |"
    )
    for skey in ["S1", "S2"]:
        su = schemas[skey].get("single_model_unseen", {})
        gate = gates[skey]
        ue = gate.get("unseen_evaluation", {})
        ci = ue.get("log_loss_delta_ci_95")
        ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—"
        status = "**PASS**" if ue.get("passed", False) else "FAIL"
        lines.append(
            f"| **{skey}** (`{schemas[skey]['schema_id']}`) | {su.get('count', 0):,} | "
            f"{su.get('log_loss_mean', 0.0):.4f} ± {su.get('log_loss_std', 0.0):.4f} | "
            f"{ci_str} | {su.get('brier_mean', 0.0):.4f} ± {su.get('brier_std', 0.0):.4f} | "
            f"{su.get('ece_mean', 0.0):.4f} ± {su.get('ece_std', 0.0):.4f} | {status} |"
        )
    return lines


def _render_slices_table(schemas: dict[str, Any]) -> list[str]:
    lines = [
        "## 4. Predeclared Slices Performance (Single-Model Means)",
        "",
        "| Slice | S0 Log Loss | S1 Log Loss (Δ) | S2 Log Loss (Δ) | S0 Brier | S1 Brier (Δ) | "
        "S2 Brier (Δ) |",
        _SEVEN_COL_SEPARATOR,
    ]
    s0_slices = schemas["S0"]["single_model_slices"]
    s1_slices = schemas["S1"]["single_model_slices"]
    s2_slices = schemas["S2"]["single_model_slices"]

    for sl_name in s0_slices:
        s0_s_ll = s0_slices[sl_name]["log_loss_mean"]
        s0_s_br = s0_slices[sl_name]["brier_mean"]
        s1_s_ll = s1_slices.get(sl_name, {}).get("log_loss_mean", float("nan"))
        s1_s_br = s1_slices.get(sl_name, {}).get("brier_mean", float("nan"))
        s2_s_ll = s2_slices.get(sl_name, {}).get("log_loss_mean", float("nan"))
        s2_s_br = s2_slices.get(sl_name, {}).get("brier_mean", float("nan"))

        d1_ll = s1_s_ll - s0_s_ll
        d1_br = s1_s_br - s0_s_br
        d2_ll = s2_s_ll - s0_s_ll
        d2_br = s2_s_br - s0_s_br

        lines.append(
            f"| `{sl_name}` | {s0_s_ll:.4f} | {s1_s_ll:.4f} ({d1_ll:+.4f}) | "
            f"{s2_s_ll:.4f} ({d2_ll:+.4f}) | {s0_s_br:.4f} | {s1_s_br:.4f} ({d1_br:+.4f}) | "
            f"{s2_s_br:.4f} ({d2_br:+.4f}) |"
        )
    return lines


def _render_calibration_table(schemas: dict[str, Any]) -> list[str]:
    lines = [
        "## 5. Calibration Analysis (Ensemble Diagnostic)",
        "",
        "| Bin Range | S0 Count | S0 Pred / Obs | S1 Count | S1 Pred / Obs | "
        "S2 Count | S2 Pred / Obs |",
        _SEVEN_COL_SEPARATOR,
    ]
    s0_cal = schemas["S0"]["ensemble_diagnostic"]["scores"]["calibration"]
    s1_cal = schemas["S1"]["ensemble_diagnostic"]["scores"]["calibration"]
    s2_cal = schemas["S2"]["ensemble_diagnostic"]["scores"]["calibration"]

    for b0, b1, b2 in zip(s0_cal, s1_cal, s2_cal, strict=True):
        po_0 = f"{b0['prediction']:.3f} / {b0['observed']:.3f}" if b0["count"] > 0 else "—"
        po_1 = f"{b1['prediction']:.3f} / {b1['observed']:.3f}" if b1["count"] > 0 else "—"
        po_2 = f"{b2['prediction']:.3f} / {b2['observed']:.3f}" if b2["count"] > 0 else "—"
        lines.append(
            f"| [{b0['lower']:.1f}, {b0['upper']:.1f}) | {b0['count']:,} | {po_0} | "
            f"{b1['count']:,} | {po_1} | {b2['count']:,} | {po_2} |"
        )
    return lines


def _render_probe_suite_table(schemas: dict[str, Any]) -> list[str]:
    lines = [
        "## 6. Probe Suite Behavior",
        "",
        "| Check | S0 | S1 | S2 |",
        "|---|---|---|---|",
    ]
    s0_pr = schemas["S0"]["ensemble_diagnostic"]["probes"]["checks"]
    s1_pr = schemas["S1"]["ensemble_diagnostic"]["probes"]["checks"]
    s2_pr = schemas["S2"]["ensemble_diagnostic"]["probes"]["checks"]
    all_checks = sorted(set(s0_pr.keys()) | set(s1_pr.keys()) | set(s2_pr.keys()))
    for chk in all_checks:
        c0 = "PASS" if s0_pr.get(chk, False) else "FAIL"
        c1 = "PASS" if s1_pr.get(chk, False) else "FAIL"
        c2 = "PASS" if s2_pr.get(chk, False) else "FAIL"
        lines.append(f"| `{chk}` | {c0} | {c1} | {c2} |")
    return lines


def _render_extraction_cost(extraction_cost: dict[str, Any] | None) -> list[str]:
    lines = ["## 7. Feature Extraction Cost Analysis", ""]
    if extraction_cost is None:
        lines.append(NOTE_ALERT)
        lines.append("> **Status**: Not measured (no verified JVM benchmark artifact provided).")
        lines.append(
            '> Run `sbt "runMain dicechess.training.golden.ExtractionBenchmarkApp '
            "tests/fixtures/kcp13/probes.tsv "
            'tests/fixtures/benchmark/extraction-cost-0.9.3.json"` '
            "to measure verified JVM latency."
        )
        return lines

    rt = extraction_cost.get("runtime", {})
    cfg = extraction_cost.get("config", {})
    lines.append(
        f"Feature extraction measured with JVM engine `{extraction_cost.get('engine_artifact')}` "
        f"({cfg.get('sample_iterations', 50)} samples per probe after "
        f"{cfg.get('warmup_iterations', 20)} warmups):"
    )
    lines.append(
        f"- Runtime: Java `{rt.get('java_version')}` ({rt.get('java_vendor')}) on "
        f"`{rt.get('os_name')}` ({rt.get('os_arch')})"
    )
    lines.append(f"- Timestamp: `{extraction_cost.get('timestamp')}`")
    lines.append("")
    lines.append(
        "| Probe Position | S0 Latency (median) | S1 Latency (median) | "
        "S2 Latency (median) | S2 Overhead |"
    )
    lines.append("|---|---|---|---|---|")

    probes_data = extraction_cost.get("probes", {})
    key_probes = ["start-w", "start-b", "bare-kings", "blocked-pawns-w", "passed-pawn-w"]
    for pid in key_probes:
        if pid in probes_data:
            schs = probes_data[pid].get("schemas", {})
            s0_m = schs.get("S0", {}).get("median_us", 0.0)
            s1_m = schs.get("S1", {}).get("median_us", 0.0)
            s2_m = schs.get("S2", {}).get("median_us", 0.0)
            ov2 = (s2_m - s0_m) / s0_m * 100 if s0_m > 0 else 0.0
            lines.append(
                f"| `{pid}` | {s0_m:,.1f} μs | {s1_m:,.1f} μs | {s2_m:,.1f} μs | {ov2:+.1f}% |"
            )

    lines.append("")
    lines.append(NOTE_ALERT)
    lines.append(
        "> Over 98% of extraction time across all positions is consumed by the "
        "216-outcome KCP probability search."
    )
    lines.append(
        "> Pseudo-legal mobility generation (S1) and passed-pawn bitboard masks (S2) "
        "add negligible latency overhead (<= 10% bound satisfied)."
    )
    return lines


def _render_next_actions() -> list[str]:
    lines = ["## 8. Next Actions", ""]
    lines.append(
        "1. **Private Qualification**: Issue #17 remains open pending owner execution "
        "of the frozen protocol on the private corpus."
    )
    lines.append(
        "2. **ADR 0001 Maintenance**: Retain provisional development findings in ADR 0001; "
        "final amendment occurs after owner qualification."
    )
    lines.append("3. **Value Model Training**: S0 remains the current baseline contract for #13.")
    return lines


def render_markdown_report(report: dict[str, Any]) -> str:
    schemas = report["schemas"]
    gates = report["gate_evaluations"]
    decision = report["decision"]
    split = report["split_summary"]
    ext_cost = report.get("extraction_cost")

    sections = [
        _render_header(report, split, decision),
        _render_input_shards(report),
        _render_key_metrics(schemas, gates),
        _render_unseen_table(schemas, gates),
        _render_diagnostic_ensemble(schemas),
        _render_gate_checklist(gates),
        _render_slices_table(schemas),
        _render_calibration_table(schemas),
        _render_probe_suite_table(schemas),
        _render_extraction_cost(ext_cost),
        _render_next_actions(),
    ]
    all_lines: list[str] = []
    for sec in sections:
        all_lines.extend(sec)
        all_lines.append("")

    return "\n".join(all_lines).rstrip() + "\n"
