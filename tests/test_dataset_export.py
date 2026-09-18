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


# --- Sealed-bundle position exclusion (#42) ---------------------------------------------------

# Distinct placements so a shard can carry genuinely different positions. The king walk keeps the
# material block identical across them, which is what the exporter cross-checks.
_PLACEMENTS = [
    "4k3/8/8/8/8/8/8/4K3",
    "4k3/8/8/8/8/8/8/K7",
    "4k3/8/8/8/8/8/8/7K",
    "3k4/8/8/8/8/8/8/4K3",
]


def _positional_shard(path: Path, games: list[str], placements: list[str]) -> Path:
    """One shard where game `g` visits `placements` in order, so positions are controllable."""
    fields = list(COLUMNS.items()) + [(col, pa.float32()) for col in kcp13.COLUMN_NAMES]
    rows_data = []
    for g_index, gid in enumerate(games):
        for ply, placement in enumerate(placements):
            side = "w" if ply % 2 == 0 else "b"
            fen = f"{placement} {side} - -"
            feats = list(kcp13.material_block(fen, side)) + [0.0] * (len(kcp13.COLUMN_NAMES) - 7)
            rows_data.append(
                [gid, ply, fen, "PPP", side, np.float32(1.0 if g_index % 2 == 0 else 0.0)]
                + [np.float32(f) for f in feats]
            )
    arrays = [pa.array([r[i] for r in rows_data], type=f[1]) for i, f in enumerate(fields)]
    metadata = {
        b"feature_schema": kcp13.SCHEMA_ID.encode(),
        b"engine_version": b"0.12.0",
        b"ruleset": kcp13.RULESET_VERSION.encode(),
        b"perspective": kcp13.PERSPECTIVE.encode(),
        b"columns": ",".join(kcp13.COLUMN_NAMES).encode(),
    }
    pq.write_table(
        pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=metadata)), str(path)
    )
    return path


@pytest.fixture
def development(tmp_path):
    """A bundle whose games visit the first two placements — the 'shared openings'."""
    shards = tmp_path / "dev-shards"
    shards.mkdir()
    _positional_shard(shards / "s.parquet", [f"dev-{i:03d}" for i in range(6)], _PLACEMENTS[:2])
    directory = tmp_path / "dev"
    build_dataset(shards, directory, DatasetConfig(engine_version="0.12.0"))
    return directory


@pytest.fixture
def final_shards(tmp_path):
    """Different games, overlapping on the first two placements and novel on the last two."""
    shards = tmp_path / "final-shards"
    shards.mkdir()
    _positional_shard(shards / "s.parquet", [f"fin-{i:03d}" for i in range(6)], _PLACEMENTS)
    return shards


def test_exclusion_drops_exactly_the_repeated_positions(tmp_path, development, final_shards):
    directory = tmp_path / "sealed"
    summary = build_dataset(
        final_shards,
        directory,
        DatasetConfig(engine_version="0.12.0", exclude_positions_from=development),
    )
    # Six games x four plies, of which the first two plies of each repeat development.
    assert summary["rows"] == 12
    assert summary["excluded_positions"]["removed_rows"] == 12
    _, dev_rows = benchmark_core.load_dataset(development)
    _, sealed_rows = benchmark_core.load_dataset(directory)
    dev_keys = {benchmark_core.position_key(r) for r in dev_rows}
    assert not {benchmark_core.position_key(r) for r in sealed_rows} & dev_keys


def test_the_sealed_bundle_satisfies_the_check_that_gates_it(tmp_path, development, final_shards):
    """The exporter's own filter and `prepare_final`'s refusal must agree."""
    directory = tmp_path / "sealed"
    build_dataset(
        final_shards,
        directory,
        DatasetConfig(engine_version="0.12.0", exclude_positions_from=development),
    )
    _, dev_rows = benchmark_core.load_dataset(development)
    _, sealed_rows = benchmark_core.load_dataset(directory)
    benchmark_core.check_sealed_disjointness(sealed_rows, dev_rows)
    seen = {benchmark_core.position_key(r) for r in dev_rows}
    assert len(seen & {benchmark_core.position_key(r) for r in sealed_rows}) == 0


def test_the_manifest_binds_the_exclusion(tmp_path, development, final_shards):
    directory = tmp_path / "sealed"
    build_dataset(
        final_shards,
        directory,
        DatasetConfig(engine_version="0.12.0", exclude_positions_from=development),
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dev_manifest, _ = benchmark_core.load_dataset(development)
    assert manifest["excluded_positions_source_sha256"] == benchmark_core.digest(dev_manifest)
    assert manifest["excluded_positions_rows"] == 12


def test_an_unfiltered_export_keeps_the_manifest_it_always_wrote(tmp_path, final_shards):
    """Published dataset digests must still reproduce, so the new keys appear only when used."""
    plain = tmp_path / "plain"
    build_dataset(final_shards, plain, DatasetConfig(engine_version="0.12.0"))
    manifest = json.loads((plain / "manifest.json").read_text(encoding="utf-8"))
    assert "excluded_positions_source_sha256" not in manifest
    assert "excluded_positions_rows" not in manifest
    assert set(manifest) == {
        "schema",
        "version",
        "perspective",
        "target",
        "kind",
        "license",
        "license_evidence_sha256",
        "rows_sha256",
        "source_sha256",
        "golden_sha256",
        "feature_schema",
        "engine_version",
        "columns",
    }


def test_excluding_everything_publishes_nothing(tmp_path, development):
    shards = tmp_path / "same-shards"
    shards.mkdir()
    _positional_shard(shards / "s.parquet", [f"other-{i:03d}" for i in range(4)], _PLACEMENTS[:2])
    directory = tmp_path / "sealed"
    with pytest.raises(DatasetError, match="every row was excluded"):
        build_dataset(
            shards,
            directory,
            DatasetConfig(engine_version="0.12.0", exclude_positions_from=development),
        )
    assert not directory.exists()


def test_an_inadmissible_exclusion_source_publishes_nothing(tmp_path, final_shards):
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    (bogus / "manifest.json").write_text('{"schema": "not-a-dataset"}', encoding="utf-8")
    directory = tmp_path / "sealed"
    with pytest.raises(DatasetError, match="not an admissible dataset"):
        build_dataset(
            final_shards,
            directory,
            DatasetConfig(engine_version="0.12.0", exclude_positions_from=bogus),
        )
    assert not directory.exists()
    with pytest.raises(DatasetError, match="not a dataset directory"):
        build_dataset(
            final_shards,
            directory,
            DatasetConfig(engine_version="0.12.0", exclude_positions_from=tmp_path / "missing"),
        )


def test_cli_exposes_the_exclusion_and_reports_what_it_removed(tmp_path, development, final_shards):
    directory = tmp_path / "sealed"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "--shards",
                str(final_shards),
                "--output",
                str(directory),
                "--engine-version",
                "0.12.0",
                "--exclude-positions-from",
                str(development),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["rows"] == 12
    assert summary["excluded_positions"]["removed_rows"] == 12


def test_a_source_that_matches_nothing_is_still_recorded(tmp_path, final_shards):
    """Filtered-with-no-overlap must not look like never-filtered.

    The metadata exists to answer one question — was this bundle constructed against development?
    — so it records the construction, not its yield. Recording only non-empty removals would make
    a bundle that overlapped nowhere indistinguishable from one nobody ever filtered.
    """
    unrelated_shards = tmp_path / "unrelated-shards"
    unrelated_shards.mkdir()
    _positional_shard(unrelated_shards / "s.parquet", ["unrelated-000"], _PLACEMENTS[2:])
    unrelated = tmp_path / "unrelated"
    build_dataset(unrelated_shards, unrelated, DatasetConfig(engine_version="0.12.0"))

    # Overlapping placements only; the source above carries the other two, so nothing matches.
    shards = tmp_path / "novel-shards"
    shards.mkdir()
    _positional_shard(shards / "s.parquet", [f"novel-{i:03d}" for i in range(4)], _PLACEMENTS[:2])
    directory = tmp_path / "sealed"
    summary = build_dataset(
        shards, directory, DatasetConfig(engine_version="0.12.0", exclude_positions_from=unrelated)
    )

    assert summary["excluded_positions"]["removed_rows"] == 0
    assert summary["rows"] == 8
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    unrelated_manifest, _ = benchmark_core.load_dataset(unrelated)
    assert manifest["excluded_positions_rows"] == 0
    assert manifest["excluded_positions_source_sha256"] == benchmark_core.digest(unrelated_manifest)

    # And the identity differs from the same rows exported without the option, which is the point.
    plain = tmp_path / "plain"
    plain_summary = build_dataset(shards, plain, DatasetConfig(engine_version="0.12.0"))
    assert plain_summary["rows_sha256"] == summary["rows_sha256"]
    assert plain_summary["manifest_sha256"] != summary["manifest_sha256"]
