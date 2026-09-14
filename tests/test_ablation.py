"""Tests for feature schema ablation protocol, runner, and shard reader."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dicechess_training.ablation.report import render_markdown_report
from dicechess_training.ablation.runner import DEFAULT_PROTOCOL_PATH, ValueMLP, predict, train_model
from dicechess_training.contracts import SCHEMA_CONTRACTS
from dicechess_training.schema import read_enriched_shard, read_enriched_shards

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_definition():
    assert DEFAULT_PROTOCOL_PATH.exists()
    protocol = json.loads(DEFAULT_PROTOCOL_PATH.read_bytes())
    assert protocol["protocol_version"] == "playground-feature-ablation-v1"
    assert protocol["engine_version"] == "0.9.3"
    assert "S0" in protocol["schemas"]
    assert "S1" in protocol["schemas"]
    assert "S2" in protocol["schemas"]

    for _skey, sinfo in protocol["schemas"].items():
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
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_render_markdown_report_mock():
    mock_report = {
        "protocol_version": "v1",
        "protocol_sha256": "abcdef1234567890abcdef1234567890",
        "engine_version": "0.9.3",
        "split_summary": {
            "total_positions": 49000,
            "decisive_positions": 48000,
            "train_positions": 38400,
            "val_positions": 9600,
            "train_games": 800,
            "val_games": 200,
        },
        "schemas": {
            "S0": {
                "schema_id": "kcp-13",
                "feature_count": 13,
                "mean_scores": {
                    "log_loss": 0.5500,
                    "brier": 0.1800,
                    "ece": 0.0200,
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
                "mean_slices": {"phase:opening": {"log_loss": 0.6000, "brier": 0.2000}},
                "probe_suite_mean": {"checks": {"opening:start-w==start-b": True}},
            },
            "S1": {
                "schema_id": "kcp-mobility-27-v1",
                "feature_count": 27,
                "mean_scores": {
                    "log_loss": 0.5400,
                    "brier": 0.1750,
                    "ece": 0.0180,
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
                "mean_slices": {"phase:opening": {"log_loss": 0.5900, "brier": 0.1950}},
                "probe_suite_mean": {"checks": {"opening:start-w==start-b": True}},
            },
            "S2": {
                "schema_id": "kcp-mobility-pawns-31-v1",
                "feature_count": 31,
                "mean_scores": {
                    "log_loss": 0.5300,
                    "brier": 0.1700,
                    "ece": 0.0170,
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
                "mean_slices": {"phase:opening": {"log_loss": 0.5800, "brier": 0.1900}},
                "probe_suite_mean": {"checks": {"opening:start-w==start-b": True}},
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
        "decision": {
            "selected_schema": "S2",
            "selected_schema_id": "kcp-mobility-pawns-31-v1",
            "gate_results": {"S1": True, "S2": True},
        },
    }
    md = render_markdown_report(mock_report)
    assert "# Feature Schema Ablation Report (Issue #17)" in md
    assert "kcp-mobility-pawns-31-v1" in md
    assert "PASSED" in md
