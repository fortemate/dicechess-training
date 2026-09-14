"""Tests for feature schema ablation protocol, runner, and shard reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dicechess_training.ablation.__main__ import _write_exclusive
from dicechess_training.ablation.report import (
    _render_extraction_cost,
    render_markdown_report,
)
from dicechess_training.ablation.runner import (
    DEFAULT_PROTOCOL_PATH,
    ValueMLP,
    _evaluate_unseen_gate,
    _load_extraction_cost,
    _paired_bootstrap_mean_estimand,
    _prepare_dataset_splits,
    _validate_cross_schema_integrity,
    _validate_schema_row_alignment,
    predict,
    train_model,
)
from dicechess_training.contracts import SCHEMA_CONTRACTS
from dicechess_training.schema import read_enriched_shard, read_enriched_shards

ROOT = Path(__file__).resolve().parents[1]


def _make_dummy_schema_df(
    keys: list[tuple[str, int]],
    schema_id: str,
    fen: str = "4k3/8/8/8/8/8/8/4K3 w - -",
    side: str = "w",
    dice: str = "PPP",
    result: float = 1.0,
    prefix_filler: float = 1.0,
) -> pd.DataFrame:
    cols = list(SCHEMA_CONTRACTS[schema_id].COLUMN_NAMES)
    data: dict[str, Any] = {
        "game_id": [k[0] for k in keys],
        "ply": [k[1] for k in keys],
        "fen": [fen] * len(keys),
        "side": [side] * len(keys),
        "dice": [dice] * len(keys),
        "result": [result] * len(keys),
    }
    for c in cols:
        data[c] = np.full(len(keys), prefix_filler, dtype=np.float32)
    return pd.DataFrame(data)


def test_protocol_definition():
    assert DEFAULT_PROTOCOL_PATH.exists()
    protocol_v2 = json.loads(DEFAULT_PROTOCOL_PATH.read_bytes())
    assert protocol_v2["protocol_version"] == "playground-feature-ablation-v2"
    assert protocol_v2["amends"] == "playground-feature-ablation-v1"
    assert protocol_v2["engine_version"] == "0.9.3"
    assert "S0" in protocol_v2["schemas"]
    assert "S1" in protocol_v2["schemas"]
    assert "S2" in protocol_v2["schemas"]

    # Historical lineage protocol v1 is preserved
    hist_v1_path = ROOT / "docs/ablation/protocol-v1.json"
    assert hist_v1_path.exists()
    protocol_v1 = json.loads(hist_v1_path.read_bytes())
    assert protocol_v1["protocol_version"] == "playground-feature-ablation-v1"

    # Protocol v2 includes 80/10/10 split, unseen gate, and extraction cost gate
    assert protocol_v2["split"]["train_cutoff"] == 8000
    assert protocol_v2["split"]["val_cutoff"] == 9000
    assert protocol_v2["split"]["test_policy"] == "reserved-holdout-excluded-from-selection"
    assert protocol_v2["split"]["leakage_audit"] == "canonical-position-exact"
    assert protocol_v2["estimand"]["primary"] == "single-model-replication"
    assert protocol_v2["estimand"]["paired_uncertainty"] == "whole-game-bootstrap-mean-seed-delta"
    assert protocol_v2["estimand"]["diagnostic_ensemble"] is True

    assert protocol_v2["gate"]["require_unseen_positions"] is True
    assert protocol_v2["gate"]["require_extraction_cost_evidence"] is True
    assert protocol_v2["gate"]["max_extraction_latency_overhead"] == 0.10

    for _skey, sinfo in protocol_v2["schemas"].items():
        sid = sinfo["schema_id"]
        assert sid in SCHEMA_CONTRACTS
        assert sinfo["feature_count"] == len(SCHEMA_CONTRACTS[sid].COLUMN_NAMES)


def test_reader_fails_closed_on_tampered_metadata(tmp_path: Path):
    # Create a parquet file with missing or mismatched metadata
    df = pd.DataFrame({"col_a": [1.0, 2.0], "col_b": [3.0, 4.0]})
    bad_file = tmp_path / "bad.parquet"
    df.to_parquet(bad_file)

    with pytest.raises(ValueError, match="shard feature schema None != supported"):
        read_enriched_shard(bad_file, "kcp-13", "0.9.3")


def test_reader_s0_shards():
    s0_dir = ROOT / "data/enriched/kcp-13"
    if not s0_dir.exists():
        pytest.skip("data/enriched/kcp-13 not yet generated")
    df = read_enriched_shards(s0_dir, "kcp-13", "0.9.3")
    assert len(df) == 49000
    assert "ply" in df.columns
    assert "p_diff" in df.columns
    assert "king_capture_attack" in df.columns


def test_reader_s1_shards():
    s1_dir = ROOT / "data/enriched/kcp-mobility-27-v1"
    if not s1_dir.exists():
        pytest.skip("data/enriched/kcp-mobility-27-v1 not yet generated")
    df = read_enriched_shards(s1_dir, "kcp-mobility-27-v1", "0.9.3")
    assert len(df) == 49000
    assert "own_moves_p" in df.columns
    assert "opp_moves_p" in df.columns
    assert "own_pdi" in df.columns


def test_reader_s2_shards():
    s2_dir = ROOT / "data/enriched/kcp-mobility-pawns-31-v1"
    if not s2_dir.exists() or len(list(s2_dir.glob("*.parquet"))) < 2:
        pytest.skip("data/enriched/kcp-mobility-pawns-31-v1 not yet generated")
    df = read_enriched_shards(s2_dir, "kcp-mobility-pawns-31-v1", "0.9.3")
    assert len(df) == 49000
    assert "own_passed_pawns" in df.columns
    assert "opp_passed_pawns" in df.columns
    assert "own_passed_max_rank" in df.columns
    assert "opp_passed_max_rank" in df.columns


def test_train_model_and_predict():
    import numpy as np

    x_train = np.random.randn(100, 13).astype(np.float32)
    y_train = np.random.choice([0.0, 1.0], size=100).astype(np.float32)
    cfg = {"hidden_dims": [16, 16], "learning_rate": 0.01, "batch_size": 32, "epochs": 2}
    model = train_model(x_train, y_train, 13, cfg, seed=42)
    assert isinstance(model, ValueMLP)

    preds = predict(model, x_train[:5])
    assert len(preds) == 5
    assert (preds >= 0.0).all()
    assert (preds <= 1.0).all()


def test_render_markdown_report_mock():
    mock_report = {
        "protocol_version": "playground-feature-ablation-v1",
        "protocol_sha256": "abcdef1234567890abcdef1234567890",
        "engine_version": "0.9.3",
        "status": "provisional-development",
        "split_summary": {
            "total_positions": 49000,
            "decisive_positions": 48000,
            "train_positions": 38400,
            "val_positions": 4800,
            "test_positions": 4800,
            "train_games": 800,
            "val_games": 100,
            "test_games": 100,
            "leakage": {"train:validation": 2, "train:test": 1, "validation:test": 0},
            "unseen_val_positions": 4000,
        },
        "input_shard_digests": {
            "S0": {"shard-0.parquet": "abcdef123456789012345678"},
            "S1": {"shard-0.parquet": "bcdefa123456789012345678"},
            "S2": {"shard-0.parquet": "cdefab123456789012345678"},
        },
        "schemas": {
            "S0": {
                "schema_id": "kcp-13",
                "feature_count": 13,
                "single_model_summary": {
                    "log_loss_mean": 0.5500,
                    "log_loss_std": 0.0100,
                    "brier_mean": 0.1800,
                    "brier_std": 0.0050,
                    "ece_mean": 0.0200,
                    "ece_std": 0.0010,
                },
                "single_model_slices": {
                    "phase:opening": {
                        "log_loss_mean": 0.6000,
                        "log_loss_std": 0.0100,
                        "brier_mean": 0.2000,
                        "brier_std": 0.0050,
                    }
                },
                "ensemble_diagnostic": {
                    "scores": {
                        "log_loss": 0.5400,
                        "brier": 0.1780,
                        "ece": 0.0190,
                        "calibration": [
                            {
                                "lower": 0.0,
                                "upper": 0.1,
                                "count": 100,
                                "prediction": 0.05,
                                "observed": 0.04,
                            }
                        ]
                        * 10,
                    },
                    "slices": {"phase:opening": {"log_loss": 0.5900, "brier": 0.1950}},
                    "probes": {"checks": {"opening:start-w==start-b": True}},
                },
            },
            "S1": {
                "schema_id": "kcp-mobility-27-v1",
                "feature_count": 27,
                "single_model_summary": {
                    "log_loss_mean": 0.5400,
                    "log_loss_std": 0.0080,
                    "brier_mean": 0.1750,
                    "brier_std": 0.0040,
                    "ece_mean": 0.0180,
                    "ece_std": 0.0010,
                },
                "single_model_slices": {
                    "phase:opening": {
                        "log_loss_mean": 0.5900,
                        "log_loss_std": 0.0080,
                        "brier_mean": 0.1950,
                        "brier_std": 0.0040,
                    }
                },
                "ensemble_diagnostic": {
                    "scores": {
                        "log_loss": 0.5350,
                        "brier": 0.1730,
                        "ece": 0.0175,
                        "calibration": [
                            {
                                "lower": 0.0,
                                "upper": 0.1,
                                "count": 100,
                                "prediction": 0.05,
                                "observed": 0.04,
                            }
                        ]
                        * 10,
                    },
                    "slices": {"phase:opening": {"log_loss": 0.5850, "brier": 0.1920}},
                    "probes": {"checks": {"opening:start-w==start-b": True}},
                },
            },
            "S2": {
                "schema_id": "kcp-mobility-pawns-31-v1",
                "feature_count": 31,
                "single_model_summary": {
                    "log_loss_mean": 0.5300,
                    "log_loss_std": 0.0070,
                    "brier_mean": 0.1700,
                    "brier_std": 0.0030,
                    "ece_mean": 0.0170,
                    "ece_std": 0.0010,
                },
                "single_model_slices": {
                    "phase:opening": {
                        "log_loss_mean": 0.5800,
                        "log_loss_std": 0.0070,
                        "brier_mean": 0.1900,
                        "brier_std": 0.0030,
                    }
                },
                "ensemble_diagnostic": {
                    "scores": {
                        "log_loss": 0.5250,
                        "brier": 0.1680,
                        "ece": 0.0165,
                        "calibration": [
                            {
                                "lower": 0.0,
                                "upper": 0.1,
                                "count": 100,
                                "prediction": 0.05,
                                "observed": 0.04,
                            }
                        ]
                        * 10,
                    },
                    "slices": {"phase:opening": {"log_loss": 0.5750, "brier": 0.1880}},
                    "probes": {"checks": {"opening:start-w==start-b": True}},
                },
            },
        },
        "gate_evaluations": {
            "S1": {
                "cleared": True,
                "relative_log_loss_gain": 0.0182,
                "log_loss_delta_ci_95": [-0.015, -0.005],
                "brier_regression": -0.005,
                "ece_regression": -0.002,
                "checks": {
                    "relative_log_loss_gain_gte_1pct": True,
                    "paired_ci_upper_lt_0": True,
                    "brier_no_regression": True,
                    "ece_regression_lte_0_01": True,
                    "slice_log_loss_lte_0_01": True,
                    "slice_brier_lte_0_01": True,
                },
            },
            "S2": {
                "cleared": True,
                "relative_log_loss_gain": 0.0364,
                "log_loss_delta_ci_95": [-0.025, -0.015],
                "brier_regression": -0.010,
                "ece_regression": -0.003,
                "checks": {
                    "relative_log_loss_gain_gte_1pct": True,
                    "paired_ci_upper_lt_0": True,
                    "brier_no_regression": True,
                    "ece_regression_lte_0_01": True,
                    "slice_log_loss_lte_0_01": True,
                    "slice_brier_lte_0_01": True,
                },
            },
        },
        "extraction_cost": None,
        "decision": {
            "status": "provisional-development",
            "selected_schema": "S2",
            "selected_schema_id": "kcp-mobility-pawns-31-v1",
            "gate_results": {"S1": True, "S2": True},
            "private_qualification_ref": (
                "Private Decision: Playground Feature Schema Qualification (Issue #17)"
            ),
        },
    }
    md = render_markdown_report(mock_report)
    assert "# Feature Schema Ablation Report (Issue #17)" in md
    assert "Provisional Development Report" in md
    assert "Private Decision: Playground Feature Schema Qualification (Issue #17)" in md
    assert "Input Shard Provenance" in md
    assert "Primary Estimand: Single-Model Replication" in md
    assert "Secondary Diagnostic: 5-Model Ensemble" in md
    assert "kcp-mobility-pawns-31-v1" in md
    assert "PASSED" in md


def test_cross_schema_integrity_validation():
    keys = [("g1", 1), ("g2", 2)]
    s0 = _make_dummy_schema_df(keys, "kcp-13")
    s1 = _make_dummy_schema_df(keys, "kcp-mobility-27-v1")
    s2 = _make_dummy_schema_df(keys, "kcp-mobility-pawns-31-v1")

    # 1. Perfectly aligned schemas pass
    _validate_cross_schema_integrity({"S0": s0, "S1": s1, "S2": s2})
    _validate_schema_row_alignment({"S0": s0, "S1": s1})

    # 2. Duplicate keys rejected
    dup_keys = [("g1", 1), ("g1", 1)]
    s0_dup = _make_dummy_schema_df(dup_keys, "kcp-13")
    with pytest.raises(ValueError, match="duplicate \\(game_id, ply\\)"):
        _validate_cross_schema_integrity({"S0": s0_dup})

    # 3. Row count mismatch rejected
    s1_short = _make_dummy_schema_df([("g1", 1)], "kcp-mobility-27-v1")
    with pytest.raises(ValueError, match="row count"):
        _validate_cross_schema_integrity({"S0": s0, "S1": s1_short})

    # 4. Missing source column rejected
    s1_no_fen = s1.drop(columns=["fen"])
    with pytest.raises(ValueError, match="missing source column 'fen'"):
        _validate_cross_schema_integrity({"S0": s0, "S1": s1_no_fen})

    # 5. Mismatched source field (result, fen, dice, side) rejected
    s1_bad_fen = _make_dummy_schema_df(keys, "kcp-mobility-27-v1", fen="8/8/8/8/8/8/8/8 w - -")
    with pytest.raises(ValueError, match="source column 'fen' does not match S0"):
        _validate_cross_schema_integrity({"S0": s0, "S1": s1_bad_fen})

    s1_bad_res = _make_dummy_schema_df(keys, "kcp-mobility-27-v1", result=0.0)
    with pytest.raises(ValueError, match="source column 'result' does not match S0"):
        _validate_cross_schema_integrity({"S0": s0, "S1": s1_bad_res})

    # 6. Float32 prefix byte-equivalence rejection
    s1_bad_prefix = s1.copy()
    s1_bad_prefix.loc[0, "p_diff"] = 99.0
    with pytest.raises(ValueError, match="Schema S1 float32 prefix does not match S0"):
        _validate_cross_schema_integrity({"S0": s0, "S1": s1_bad_prefix})

    s2_bad_prefix13 = s2.copy()
    s2_bad_prefix13.loc[0, "p_diff"] = 99.0
    with pytest.raises(ValueError, match="Schema S2 float32 13-feature prefix does not match S0"):
        _validate_cross_schema_integrity({"S0": s0, "S1": s1, "S2": s2_bad_prefix13})

    s2_bad_prefix27 = s2.copy()
    s2_bad_prefix27.loc[0, "own_moves_p"] = 99.0
    with pytest.raises(ValueError, match="Schema S2 float32 27-feature prefix does not match S1"):
        _validate_cross_schema_integrity({"S0": s0, "S1": s1, "S2": s2_bad_prefix27})


def test_prepare_dataset_splits_80_10_10_and_leakage_audit():
    # game_1 hashes to 5560 (< 8000 -> train)
    # game_0 hashes to 8185 (>= 8000 & < 9000 -> val)
    # game_4 hashes to 9151 (>= 9000 -> test holdout)
    fen_shared = "4k3/8/8/8/8/8/8/4K3 w - -"
    fen_val_unique = "4k3/8/8/8/8/8/8/4K3 b - -"

    rows = [
        {"game_id": "game_1", "ply": 1, "fen": fen_shared, "result": 1.0},
        {"game_id": "game_0", "ply": 1, "fen": fen_shared, "result": 1.0},  # overlaps with train
        {"game_id": "game_0", "ply": 2, "fen": fen_val_unique, "result": 1.0},  # unique to val
        {"game_id": "game_4", "ply": 1, "fen": fen_shared, "result": 1.0},  # test holdout
        {"game_id": "game_1", "ply": 2, "fen": "8/8/8/8/8/8/8/8 w - -", "result": 0.5},  # draw
    ]
    df = pd.DataFrame(rows)
    protocol = {
        "split": {
            "train_cutoff": 8000,
            "val_cutoff": 9000,
            "draw_policy": "exclude-from-binary-metrics",
        }
    }

    (
        train_mask,
        val_mask,
        val_df,
        y_val,
        groups_val,
        summary,
        unseen_val_mask,
    ) = _prepare_dataset_splits(df, protocol)

    # 1. Verification of masks
    assert train_mask.sum() == 1  # only decisive game_1 row
    assert val_mask.sum() == 2  # game_0 rows (hash 8185)
    assert summary["test_positions"] == 1  # game_4 row (hash 9151) excluded from val
    assert len(val_df) == 2
    assert (val_df["game_id"] == "game_0").all()

    # 2. Verification of position leakage audit
    leak = summary["leakage"]
    assert leak["train:validation"] == 1  # fen_shared is in both train and val
    assert leak["train:test"] == 1  # fen_shared is in both train and test
    assert summary["unseen_val_positions"] == 1  # fen_val_unique is unseen in train
    assert unseen_val_mask.sum() == 1


def test_cli_write_exclusive(tmp_path: Path):
    target = tmp_path / "report.md"
    _write_exclusive(target, "initial content", overwrite=False)
    assert target.read_text() == "initial content"

    # Exclusive write fails when file exists and overwrite=False
    with pytest.raises(FileExistsError, match="Destination file already exists"):
        _write_exclusive(target, "new content", overwrite=False)

    # Overwrite allowed when explicit flag is True
    _write_exclusive(target, "updated content", overwrite=True)
    assert target.read_text() == "updated content"


def test_extraction_cost_loading_and_rendering():
    protocol = json.loads(DEFAULT_PROTOCOL_PATH.read_bytes())
    fixture_path = ROOT / "tests/fixtures/benchmark/extraction-cost-0.9.3.json"

    # None and missing path return None
    assert _load_extraction_cost(None, protocol) is None
    assert _load_extraction_cost(ROOT / "nonexistent.json", protocol) is None

    # Valid fixture loads successfully
    loaded = _load_extraction_cost(fixture_path, protocol)
    assert loaded is not None
    assert loaded["schema"] == "playground-extraction-benchmark-v1"
    assert loaded["engine_version"] == "0.9.3"
    assert "start-w" in loaded["probes"]

    # Invalid engine version raises ValueError
    bad_protocol = {**protocol, "engine_version": "0.9.2"}
    with pytest.raises(ValueError, match="Extraction benchmark engine"):
        _load_extraction_cost(fixture_path, bad_protocol)

    # Rendering with None produces "Not measured"
    unmeasured_lines = _render_extraction_cost(None)
    assert any("Not measured" in line for line in unmeasured_lines)

    # Rendering with verified data produces latency table
    measured_lines = _render_extraction_cost(loaded)
    text = "\n".join(measured_lines)
    assert "com.fortemate:dicechess-engine_3:0.9.3" in text
    assert "`start-w`" in text
    assert "S2 Overhead" in text


def test_reader_fails_closed_on_non_float32_features(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [
        pa.field("game_id", pa.string()),
        pa.field("ply", pa.int32()),
        pa.field("fen", pa.string()),
        pa.field("dice", pa.string()),
        pa.field("side", pa.string()),
        pa.field("result", pa.float32()),
    ]
    data: dict[str, list] = {
        "game_id": ["g1"],
        "ply": [1],
        "fen": ["4k3/8/8/8/8/8/8/4K3 w - -"],
        "dice": ["PPP"],
        "side": ["w"],
        "result": [1.0],
    }
    for name in SCHEMA_CONTRACTS["kcp-13"].COLUMN_NAMES:
        fields.append(pa.field(name, pa.float64()))
        data[name] = [1.0]

    meta = {
        b"feature_schema": b"kcp-13",
        b"feature_count": b"13",
        b"engine_version": b"0.9.3",
        b"ruleset": b"standard-dicechess-v1",
        b"perspective": b"side-to-move",
    }
    schema = pa.schema(fields, metadata=meta)
    arrays = [pa.array(data[f.name], type=f.type) for f in fields]
    table = pa.Table.from_arrays(arrays, schema=schema)
    test_path = tmp_path / "bad_type.parquet"
    pq.write_table(table, test_path)

    with pytest.raises(ValueError, match="has type double, expected float"):
        read_enriched_shard(test_path, "kcp-13", "0.9.3")


def test_paired_bootstrap_mean_estimand_complementary_errors():
    # 4 games with 2 rows each = 8 rows
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])

    # Baseline prediction = 0.5
    p_ref = np.full(8, 0.5, dtype=np.float32)

    # Seed 0 has error delta -d on every row (wins lose less, losses lose less)
    # Seed 1 has complementary error delta +d on every row
    # For y=1: p0 = 0.6 -> ll_0 = -ln(0.6) = 0.5108256; ref_ll = ln(2) = 0.6931472
    #          delta = -0.1823216
    #          p1 = exp(-(ln(2) - (-0.1823216))) = 0.41666666 -> delta = +0.1823216
    # For y=0: p0 = 0.4 -> ll_0 = -ln(0.6) = 0.5108256 -> delta = -0.1823216
    #          p1 = 1 - 0.41666666 = 0.5833333 -> delta = +0.1823216
    p_s0 = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4], dtype=np.float32)
    p_s1 = np.array(
        [
            0.41666666,
            0.5833333,
            0.41666666,
            0.5833333,
            0.41666666,
            0.5833333,
            0.41666666,
            0.5833333,
        ],
        dtype=np.float32,
    )

    res = _paired_bootstrap_mean_estimand(
        y=y,
        s_preds=[p_s0, p_s1],
        s0_preds=[p_ref, p_ref],
        groups=groups,
        repeats=100,
        seed=13,
    )

    # The mean loss delta across seeds for every row is identically 0.0
    # Therefore, the mean-seed bootstrap CI collapses to [0.0, 0.0] (degenerate)
    ci = res["log_loss_delta_ci_95"]
    assert pytest.approx(ci[0], abs=1e-5) == 0.0
    assert pytest.approx(ci[1], abs=1e-5) == 0.0

    # In contrast, per-seed individual CIs are non-degenerate
    seed_cis = res["per_seed_log_loss_delta_cis"]
    assert len(seed_cis) == 2
    assert seed_cis[0][1] < 0.0  # Seed 0 is strictly negative (improved)
    assert seed_cis[1][0] > 0.0  # Seed 1 is strictly positive (regressed)


def test_unseen_gate_fails_on_unseen_regression():
    protocol = {
        "uncertainty": {"repeats": 50, "seed": 13},
        "gate": {
            "min_relative_log_loss_gain": 0.01,
            "require_negative_paired_log_loss_ci_upper": True,
            "max_brier_regression": 0.0,
            "max_ece_regression": 0.01,
            "max_slice_log_loss_regression": 0.01,
            "max_slice_brier_regression": 0.01,
            "require_unseen_positions": True,
            "max_unseen_log_loss_regression": 0.01,
            "max_unseen_brier_regression": 0.0,
            "require_extraction_cost_evidence": True,
            "max_extraction_latency_overhead": 0.10,
        },
    }

    # Full validation has 4 rows, unseen mask is True for 2 rows
    unseen_val_mask = np.array([True, True, False, False])
    y_val = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    groups_val = np.array(["g1", "g1", "g2", "g2"])

    p_s0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    # Candidate improves on rows 2,3 but regresses on unseen rows 0,1
    p_s1 = np.array([0.3, 0.7, 0.9, 0.1], dtype=np.float32)

    results_by_schema = {
        "S0": {
            "single_model_summary": {"log_loss_mean": 0.70, "brier_mean": 0.25, "ece_mean": 0.05},
            "single_model_slices": {},
            "single_model_unseen": {"log_loss_mean": 0.693, "brier_mean": 0.25},
        },
        "S1": {
            "single_model_summary": {"log_loss_mean": 0.50, "brier_mean": 0.20, "ece_mean": 0.03},
            # Severe unseen regression:
            "single_model_unseen": {"log_loss_mean": 1.200, "brier_mean": 0.35},
        },
    }

    val_preds_by_schema = {"S0": [p_s0], "S1": [p_s1]}

    unseen_eval = _evaluate_unseen_gate(
        s_key="S1",
        results_by_schema=results_by_schema,
        val_preds_by_schema=val_preds_by_schema,
        y_val=y_val,
        groups_val=groups_val,
        unseen_val_mask=unseen_val_mask,
        protocol=protocol,
    )

    assert unseen_eval["passed"] is False
    assert unseen_eval["passed_ll"] is False
    assert unseen_eval["reason"] == "unseen-metrics-regressed"


def test_unseen_gate_fails_on_empty_unseen():
    protocol = {
        "uncertainty": {"repeats": 10, "seed": 13},
        "gate": {
            "max_unseen_log_loss_regression": 0.01,
            "max_unseen_brier_regression": 0.0,
        },
    }
    unseen_val_mask = np.array([False, False, False])
    unseen_eval = _evaluate_unseen_gate(
        s_key="S1",
        results_by_schema={},
        val_preds_by_schema={},
        y_val=np.array([1.0, 0.0, 1.0]),
        groups_val=np.array(["g1", "g1", "g2"]),
        unseen_val_mask=unseen_val_mask,
        protocol=protocol,
    )
    assert unseen_eval["passed"] is False
    assert unseen_eval["reason"] == "no-unseen-positions"


def test_paired_bootstrap_cancels_complementary_errors_across_groups():
    # Each seed varies across groups; their mean loss delta is constant per row.
    y = np.ones(20)
    a = np.array([0.2] * 10 + [0.8] * 10)
    reference = np.full(20, 0.5)
    result = _paired_bootstrap_mean_estimand(
        y, [a, a[::-1]], [reference, reference], np.arange(20), repeats=1000, seed=13
    )
    expected = np.log(1.25)
    np.testing.assert_allclose(result["log_loss_delta_ci_95"], [expected, expected])
    for lower, upper in result["per_seed_log_loss_delta_cis"]:
        assert lower < upper
