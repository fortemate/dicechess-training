"""Characterize synthetic bundle bytes, validation order and failed publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from dicechess_training.contracts import kcp13
from dicechess_training.dataset import DatasetConfig, DatasetError, build_dataset
from dicechess_training.dataset import export as exporter
from test_dataset_export import _PLACEMENTS, _positional_shard

GOLDENS = Path(__file__).parent / "fixtures/dataset-export-format.json"


def _source(folder, form):
    """Authored king-walk rows; no production games, labels or trained results."""
    folder.mkdir()
    if form == "reversed-list":
        first = _positional_shard(folder / "a.parquet", ["gamma", "alpha"], _PLACEMENTS)
        second = _positional_shard(folder / "b.parquet", ["beta", "delta"], _PLACEMENTS)
        return [second, first]
    path = _positional_shard(
        folder / "source.parquet", ["gamma", "alpha", "beta", "delta"], _PLACEMENTS
    )
    if form == "dataframe":
        frame = pq.read_table(path).to_pandas()
        frame.index = np.arange(len(frame)) * 3 + 17
        return frame
    return {"path": path, "string": str(path), "directory": folder, "list": [path]}[form]


def _synthetic_config(**options):
    return DatasetConfig(
        kind="synthetic", engine_version="0.12.0", license_name="Authored fixture", **options
    )


def _file_hashes(folder):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.iterdir())
    }


@pytest.mark.parametrize(
    "form", ["dataframe", "path", "string", "directory", "list", "reversed-list"]
)
def test_complete_synthetic_bundle_bytes_summary_and_source_immutability(tmp_path, form):
    inputs = tmp_path / "inputs"
    source = _source(inputs, form)
    original_files = _file_hashes(inputs)
    original_frame = source.copy(deep=True) if isinstance(source, pd.DataFrame) else None
    destination = tmp_path / "bundle"
    summary = build_dataset(source, destination, _synthetic_config())
    expected = json.loads(GOLDENS.read_text())[form]
    assert summary == expected["summary"]
    assert _file_hashes(destination) == expected["files_sha256"]
    assert _file_hashes(inputs) == original_files
    if original_frame is not None:
        pd.testing.assert_frame_equal(source, original_frame)
    rows = json.loads((destination / "rows.json").read_text())
    assert [row["id"] for row in rows] == [
        f"{game}:{ply}" for game in ("alpha", "beta", "delta", "gamma") for ply in range(4)
    ]


@pytest.mark.parametrize("form", ["dataframe", "path"])
def test_implicit_engine_version_uses_source_metadata_or_dataframe_default(tmp_path, form):
    source = _source(tmp_path / "inputs", form)
    config = DatasetConfig(kind="synthetic", license_name="Authored fixture")
    summary = build_dataset(source, tmp_path / "bundle", config)
    expected = kcp13.GOLDEN_ENGINE_VERSION if form == "dataframe" else "0.12.0"
    assert summary["engine_version"] == expected


def test_game_limit_selects_encounter_order_before_stable_row_sort(tmp_path):
    source = _source(tmp_path / "inputs", "dataframe")
    original = source.copy(deep=True)
    destination = tmp_path / "bundle"
    summary = build_dataset(source, destination, _synthetic_config(max_games=3))
    rows = json.loads((destination / "rows.json").read_text())
    assert [row["id"] for row in rows] == [
        f"{game}:{ply}" for game in ("alpha", "beta", "gamma") for ply in range(4)
    ]
    assert summary["games"] == 3
    assert summary["rows"] == 12
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("form", ["dataframe", "path", "directory", "list", "reversed-list"])
@pytest.mark.parametrize("limit", [1, 3, 4, 9])
def test_game_limits_preserve_complete_bundles_and_input_for_each_source_form(
    tmp_path, form, limit
):
    inputs = tmp_path / "inputs"
    source = _source(inputs, form)
    original_files = _file_hashes(inputs)
    original_frame = source.copy(deep=True) if isinstance(source, pd.DataFrame) else None
    encounter_order = (
        ["beta", "delta", "gamma", "alpha"]
        if form == "reversed-list"
        else ["gamma", "alpha", "beta", "delta"]
    )
    expected_games = encounter_order[:limit]
    # Explicit synthetic provenance lets source forms share a reference manifest.
    # The reference is prefiltered and never exercises the max_games branch.
    reference_source = _source(tmp_path / "reference-inputs", "dataframe")
    reference_source = reference_source[reference_source["game_id"].isin(expected_games)]
    reference_config = _synthetic_config(source_sha256="a" * 64)
    reference = tmp_path / "reference"
    expected = build_dataset(reference_source, reference, reference_config)
    config = _synthetic_config(max_games=limit, source_sha256="a" * 64)
    destination = tmp_path / "bundle"
    actual = build_dataset(source, destination, config)
    assert actual == expected
    assert _file_hashes(destination) == _file_hashes(reference)
    assert _file_hashes(inputs) == original_files
    if original_frame is not None:
        pd.testing.assert_frame_equal(source, original_frame)


@pytest.mark.parametrize("choice", ["file", "text", "owner-default", "synthetic-default"])
def test_license_precedence_and_exact_unicode_bytes(tmp_path, choice):
    source = _source(tmp_path / "inputs", "dataframe")
    license_file = tmp_path / "authored-license.txt"
    file_text = "Authored synthetic license\nΩ\n"
    license_file.write_text(file_text, encoding="utf-8")
    text = "Authored inline license\n終"
    options, expected = {
        "file": (
            {"kind": "public-licensed", "license_file": license_file, "license_text": text},
            file_text,
        ),
        "text": ({"kind": "public-licensed", "license_text": text}, text),
        "owner-default": ({"kind": "owner-controlled"}, exporter.DEFAULT_OWNER_LICENSE),
        "synthetic-default": ({"kind": "synthetic"}, exporter.DEFAULT_SYNTHETIC_LICENSE),
    }[choice]
    config = DatasetConfig(engine_version="0.12.0", **options)
    destination = tmp_path / "bundle"
    summary = build_dataset(source, destination, config)
    assert (destination / "license.txt").read_bytes() == expected.encode("utf-8")
    assert (
        summary["license_evidence_sha256"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"kind": "unknown", "version": "bad"}, "unsupported dataset kind"),
        ({"version": "bad", "feature_schema": "unknown"}, "invalid dataset semantic version"),
        ({"feature_schema": "unknown"}, "unsupported schema"),
    ],
)
def test_configuration_validation_precedes_existing_destination_and_source(
    tmp_path, options, message
):
    destination = tmp_path / "existing"
    destination.mkdir()
    config = DatasetConfig(**options)
    source = object()
    with pytest.raises(DatasetError, match=message):
        build_dataset(source, destination, config)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("kind", ["empty-list", "unsupported", "missing-file"])
def test_invalid_source_precedes_engine_version_validation(tmp_path, kind):
    source, message = {
        "empty-list": ([], "shards list is empty"),
        "unsupported": ((), "unsupported shards source type"),
        "missing-file": (tmp_path / "missing.parquet", "source path does not exist"),
    }[kind]
    config = DatasetConfig(engine_version="bad")
    destination = tmp_path / "bundle"
    with pytest.raises(DatasetError, match=message):
        build_dataset(source, destination, config)
    assert not destination.exists()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({kcp13.COLUMN_NAMES[0]: np.nan, "fen": "invalid"}, "feature matrix contains NaN"),
        ({"fen": "invalid", "side": "x", "ply": -1, "result": 2}, "invalid canonical FEN"),
        ({"side": "b", "ply": -1, "result": 2}, "does not match FEN active color"),
        ({"ply": -1, "result": 2}, "ply must be non-negative"),
        ({"result": 2, "fen": "4k3/8/8/8/8/8/8/3KK3 w - -"}, "invalid game result"),
        ({"fen": "4k3/8/8/8/8/8/8/3KK3 w - -"}, "invalid king count"),
        ({kcp13.COLUMN_NAMES[0]: 99}, "material block mismatch"),
    ],
)
def test_row_validation_order_and_no_publication(tmp_path, changes, message):
    source = _source(tmp_path / "inputs", "dataframe")
    for column, value in changes.items():
        source.loc[source.index[0], column] = value
    original = source.copy(deep=True)
    config = _synthetic_config()
    destination = tmp_path / "bundle"
    with pytest.raises(DatasetError, match=message):
        build_dataset(source, destination, config)
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("failure", ["consumer", "rename"])
def test_failed_validation_or_publication_removes_the_complete_staged_bundle(
    tmp_path, monkeypatch, failure
):
    source = _source(tmp_path / "inputs", "dataframe")
    destination = tmp_path / "bundle"
    staging = tmp_path / f".bundle.staging-{os.getpid()}"
    staging.mkdir()
    (staging / "stale.txt").write_text("stale prior export")
    error = OSError("synthetic publication refusal")
    observed = []

    def refuse(path, *args, **kwargs):
        observed.append(path)
        assert path == staging
        assert set(_file_hashes(path)) == {"manifest.json", "rows.json", "license.txt"}
        raise error

    if failure == "consumer":
        monkeypatch.setattr(exporter.benchmark_core, "load_dataset", refuse)
    else:
        monkeypatch.setattr(Path, "rename", refuse)
    config = _synthetic_config()
    with pytest.raises(OSError) as caught:
        build_dataset(source, destination, config)
    assert caught.value is error
    assert observed == [staging]
    assert not staging.exists()
    assert not destination.exists()


@pytest.mark.parametrize("choice", ["missing-file", "missing-text"])
def test_license_refusal_never_creates_staging_and_does_not_fall_back(tmp_path, choice):
    source = _source(tmp_path / "inputs", "dataframe")
    options, message = {
        "missing-file": (
            {"license_file": tmp_path / "missing.txt", "license_text": "unused"},
            "license file not found",
        ),
        "missing-text": ({"kind": "public-licensed"}, "license text or license file is required"),
    }[choice]
    config = DatasetConfig(engine_version="0.12.0", **options)
    destination = tmp_path / "bundle"
    with pytest.raises(DatasetError, match=message):
        build_dataset(source, destination, config)
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))
