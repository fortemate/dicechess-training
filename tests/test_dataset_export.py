"""Tests for dataset packaging and export into playground-rows-v1 format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.candidate import CandidateConfig, build_candidate
from dicechess_training.contracts import kcp13
from dicechess_training.dataset import DatasetConfig, DatasetError, build_dataset
from dicechess_training.dataset.__main__ import main
from dicechess_training.schema import COLUMNS


def _make_test_parquet(
    path: Path,
    num_games: int = 5,
    plies_per_game: int = 4,
    engine_ver: str = "0.12.0",
) -> Path:
    """Helper to write a valid enriched parquet shard for testing."""
    fields = list(COLUMNS.items()) + [(col, pa.float32()) for col in kcp13.COLUMN_NAMES]

    fen_white = "4k3/8/8/8/8/8/8/4K3 w - -"
    fen_black = "4k3/8/8/8/8/8/8/4K3 b - -"
    mat_w = kcp13.material_block(fen_white, "w")
    mat_b = kcp13.material_block(fen_black, "b")

    rows_data = []
    for g in range(num_games):
        gid = f"game-{g:04d}"
        for p in range(plies_per_game):
            side = "w" if p % 2 == 0 else "b"
            fen = fen_white if side == "w" else fen_black
            mat = mat_w if side == "w" else mat_b
            res = 1.0 if g % 2 == 0 else 0.0
            feats = list(mat) + [0.0] * (len(kcp13.COLUMN_NAMES) - 7)
            row = [gid, p, fen, "PPP", side, np.float32(res)] + [np.float32(f) for f in feats]
            rows_data.append(row)

    arrays = [
        pa.array([r[col_idx] for r in rows_data], type=f[1]) for col_idx, f in enumerate(fields)
    ]
    metadata = {
        b"feature_schema": kcp13.SCHEMA_ID.encode(),
        b"engine_version": engine_ver.encode(),
        b"ruleset": kcp13.RULESET_VERSION.encode(),
        b"perspective": kcp13.PERSPECTIVE.encode(),
        b"columns": ",".join(kcp13.COLUMN_NAMES).encode(),
    }
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=metadata))
    pq.write_table(table, str(path))
    return path


def test_build_dataset_end_to_end(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    _make_test_parquet(
        shards_dir / "shard-0.parquet", num_games=10, plies_per_game=4, engine_ver="0.12.0"
    )

    dataset_dir = tmp_path / "dataset"
    summary = build_dataset(
        shards_dir,
        dataset_dir,
        DatasetConfig(kind="owner-controlled", engine_version="0.12.0"),
    )

    assert dataset_dir.is_dir()
    assert (dataset_dir / "manifest.json").is_file()
    assert (dataset_dir / "rows.json").is_file()
    assert (dataset_dir / "license.txt").is_file()

    assert summary["rows"] == 40
    assert summary["games"] == 10
    assert summary["engine_version"] == "0.12.0"

    # Verify that benchmark loader admits the dataset without error
    manifest, rows = benchmark_core.load_dataset(dataset_dir)
    assert manifest["schema"] == "playground-rows-v1"
    assert manifest["engine_version"] == "0.12.0"
    assert len(rows) == 40

    # Verify candidate can train on this dataset
    candidate_dir = tmp_path / "candidate"
    cand_summary = build_candidate(
        dataset_dir,
        candidate_dir,
        CandidateConfig(seed=42, model_id="candidate-test-model"),
    )
    assert cand_summary["training_rows"] > 0
    assert (candidate_dir / "model.onnx").is_file()


def test_build_dataset_with_max_games(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    _make_test_parquet(shards_dir / "shard-0.parquet", num_games=10, plies_per_game=2)

    dataset_dir = tmp_path / "dataset"
    summary = build_dataset(
        shards_dir,
        dataset_dir,
        DatasetConfig(max_games=3, engine_version="0.12.0"),
    )

    assert summary["games"] == 3
    assert summary["rows"] == 6
    manifest, rows = benchmark_core.load_dataset(dataset_dir)
    assert len({r["game_id"] for r in rows}) == 3


def test_build_dataset_refuses_existing_destination(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    _make_test_parquet(shards_dir / "shard-0.parquet", num_games=2)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    with pytest.raises(DatasetError, match="already exists"):
        build_dataset(shards_dir, dataset_dir)


def test_build_dataset_refuses_unverified_engine(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    _make_test_parquet(shards_dir / "shard-0.parquet", num_games=2, engine_ver="9.9.9")

    dataset_dir = tmp_path / "dataset"
    with pytest.raises(DatasetError, match="no committed golden corpus"):
        build_dataset(shards_dir, dataset_dir, DatasetConfig(engine_version="9.9.9"))


def test_build_dataset_refuses_material_mismatch(tmp_path):
    # Corrupt material diff feature
    fields = list(COLUMNS.items()) + [(col, pa.float32()) for col in kcp13.COLUMN_NAMES]
    fen = "4k3/8/8/8/8/8/8/4K3 w - -"
    row = ["g1", 0, fen, "PPP", "w", np.float32(1.0)] + [np.float32(99.0)] + [np.float32(0)] * 12
    metadata = {
        b"feature_schema": kcp13.SCHEMA_ID.encode(),
        b"engine_version": b"0.12.0",
        b"ruleset": kcp13.RULESET_VERSION.encode(),
        b"perspective": kcp13.PERSPECTIVE.encode(),
        b"columns": ",".join(kcp13.COLUMN_NAMES).encode(),
    }
    table = pa.Table.from_arrays(
        [pa.array([row[idx]], type=f[1]) for idx, f in enumerate(fields)],
        schema=pa.schema(fields, metadata=metadata),
    )
    shard_path = tmp_path / "corrupt.parquet"
    pq.write_table(table, str(shard_path))

    dataset_dir = tmp_path / "dataset"
    with pytest.raises(DatasetError, match="material block mismatch"):
        build_dataset(shard_path, dataset_dir, DatasetConfig(engine_version="0.12.0"))


def test_cli_export_success(tmp_path, capsys):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    _make_test_parquet(shards_dir / "shard-0.parquet", num_games=4, plies_per_game=2)

    dataset_dir = tmp_path / "dataset"
    report_path = tmp_path / "report.json"

    ret = main(
        [
            "--shards",
            str(shards_dir),
            "--output",
            str(dataset_dir),
            "--engine-version",
            "0.12.0",
            "--report",
            str(report_path),
        ]
    )

    assert ret == 0
    assert dataset_dir.is_dir()
    assert report_path.is_file()
    summary = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["rows"] == 8
    assert summary["games"] == 4


def test_cli_refuses_invalid_arguments():
    assert main([]) != 0
