"""Export enriched feature shards into a benchmark-admissible dataset package.

Produces a directory conforming to the `playground-rows-v1` schema required by the benchmark
and candidate packager:
- `manifest.json`: metadata, schema, engine version, and SHA-256 digests.
- `rows.json`: deterministic array of position rows with canonical FENs and features.
- `license.txt`: data license or owner-controlled evidence text.

Staged beside the destination and published atomically by rename, so a failed or interrupted
export writes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.benchmark.splits import assignments, position_key
from dicechess_training.contracts import kcp13
from dicechess_training.schema import (
    read_enriched_shard,
    shard_content_digest,
)

MANIFEST_FILE = "manifest.json"
ROWS_FILE = "rows.json"
LICENSE_FILE = "license.txt"

DEFAULT_OWNER_LICENSE = """Fortemate internal evaluation dataset.
Owner-controlled private data policy under dc-shared:publication v1.
Not approved for public redistribution.
"""

DEFAULT_SYNTHETIC_LICENSE = """Authored synthetic test fixture by Fortemate.
AGPL-3.0-only; see repository LICENSE.
"""


class DatasetError(ValueError):
    """A dataset could not be packaged or verified. Messages never carry private paths."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


@dataclass(frozen=True)
class DatasetConfig:
    """Settings controlling dataset packaging and provenance."""

    kind: str = "owner-controlled"
    version: str = "1.0.0"
    license_name: str = "Fortemate Owner-Controlled Data Policy v1"
    license_text: str | None = None
    license_file: Path | str | None = None
    engine_version: str | None = None
    feature_schema: str = kcp13.SCHEMA_ID
    max_games: int | None = None
    source_sha256: str | None = None
    #: An existing bundle whose exact positions must not appear in this one. Sealed final
    #: qualification refuses any exact-position overlap with development, and openings are shared
    #: across games, so a bundle drawn from disjoint games still collides on its early plies —
    #: every game contains the initial position. Leave unset for a development bundle: an export
    #: without exclusion writes the same manifest it always did, so published digests still
    #: reproduce.
    exclude_positions_from: Path | str | None = None


def _detect_engine_version(shard_path: Path) -> str:
    """Extract engine_version from the shard parquet metadata."""
    try:
        metadata = pq.read_metadata(str(shard_path)).metadata or {}
    except Exception as e:
        raise DatasetError(f"cannot read parquet metadata from shard: {e}") from e
    engine_ver = metadata.get(b"engine_version")
    if engine_ver is None:
        engine_ver = metadata.get("engine_version")
    if engine_ver is None:
        raise DatasetError("shard metadata missing engine_version")
    ver_str = engine_ver.decode("utf-8") if isinstance(engine_ver, bytes) else str(engine_ver)
    return ver_str.strip()


def _compute_source_digest(paths: list[Path]) -> str:
    """Compute a deterministic digest over the input shard contents."""
    digests = [shard_content_digest(p) for p in sorted(paths)]
    h = hashlib.sha256()
    for d in digests:
        h.update(d.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _load_shards(
    paths: list[Path],
    expected_schema: str,
    expected_engine: str | None,
) -> pd.DataFrame:
    """Read shards in order, failing closed on metadata or column mismatch."""
    frames = [read_enriched_shard(p, expected_schema, expected_engine) for p in paths]
    return pd.concat(frames, ignore_index=True)


def build_dataset(
    shards_source: str | Path | list[Path] | pd.DataFrame,
    output_dir: str | Path,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    """Package enriched positions into an admitted `playground-rows-v1` bundle.

    Writes manifest.json, rows.json, and license.txt atomically to output_dir, and validates
    with `benchmark.core.load_dataset` before completion.
    """
    cfg = config or DatasetConfig()
    _require(
        cfg.kind in ("owner-controlled", "public-licensed", "synthetic"),
        f"unsupported dataset kind: {cfg.kind!r}",
    )
    _require(
        re.fullmatch(r"\d+\.\d+\.\d+", cfg.version, flags=re.ASCII) is not None,
        f"invalid dataset semantic version: {cfg.version!r}",
    )
    _require(cfg.feature_schema == kcp13.SCHEMA_ID, f"unsupported schema {cfg.feature_schema!r}")

    out_path = Path(output_dir).resolve()
    _require(not out_path.is_symlink(), "output path must not be a symlink")
    _require(not out_path.exists(), f"output destination already exists: {out_path.name}")

    # Determine input shards and engine version
    input_paths: list[Path] = []
    engine_ver = cfg.engine_version

    if isinstance(shards_source, (str, Path)):
        src = Path(shards_source).resolve()
        _require(src.exists(), f"source path does not exist: {src}")
        if src.is_dir():
            input_paths = sorted(src.glob("*.parquet"))
            _require(bool(input_paths), f"no parquet shards found in {src}")
        elif src.is_file():
            input_paths = [src]
        else:
            raise DatasetError("source must be a file or directory")
        if engine_ver is None:
            engine_ver = _detect_engine_version(input_paths[0])
        df = _load_shards(input_paths, cfg.feature_schema, engine_ver)
    elif isinstance(shards_source, list):
        _require(bool(shards_source), "shards list is empty")
        input_paths = [Path(p).resolve() for p in shards_source]
        for p in input_paths:
            _require(p.is_file(), f"shard file does not exist: {p.name}")
        if engine_ver is None:
            engine_ver = _detect_engine_version(input_paths[0])
        df = _load_shards(input_paths, cfg.feature_schema, engine_ver)
    elif isinstance(shards_source, pd.DataFrame):
        df = shards_source.copy()
        if engine_ver is None:
            engine_ver = kcp13.GOLDEN_ENGINE_VERSION
    else:
        raise DatasetError(f"unsupported shards source type: {type(shards_source)}")

    _require(
        isinstance(engine_ver, str)
        and re.fullmatch(r"\d+\.\d+\.\d+", engine_ver, flags=re.ASCII) is not None,
        f"invalid engine version: {engine_ver!r}",
    )

    # Verify that engine has a committed golden fixture
    golden_path = kcp13.golden_path(engine_ver)
    _require(
        golden_path.is_file(),
        f"no committed golden corpus fixture found for engine version {engine_ver}",
    )
    golden_sha256 = kcp13.sha256_of(golden_path)

    # Filter by max_games if specified
    if cfg.max_games is not None:
        _require(cfg.max_games > 0, "max_games must be positive")
        unique_games = df["game_id"].drop_duplicates()
        if len(unique_games) > cfg.max_games:
            selected_games = set(unique_games.iloc[: cfg.max_games])
            df = df[df["game_id"].isin(selected_games)].copy()

    _require(len(df) > 0, "dataset contains no rows")

    # Sort stably by game_id and ply
    df = df.sort_values(["game_id", "ply"], kind="stable").reset_index(drop=True)

    # Format rows
    feature_cols = list(kcp13.COLUMN_NAMES)
    for col in feature_cols:
        _require(col in df.columns, f"missing expected feature column {col!r}")

    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    _require(np.isfinite(feature_matrix).all(), "feature matrix contains NaN or Inf values")

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        gid = str(row["game_id"])
        ply = int(row["ply"])
        fen = str(row["fen"])
        side = str(row["side"])
        result = float(row["result"])

        fields = fen.split()
        _require(len(fields) in (4, 6) and fields[1] in ("w", "b"), f"invalid canonical FEN: {fen}")
        _require(side == fields[1], f"side {side!r} does not match FEN active color {fields[1]!r}")
        _require(ply >= 0, f"ply must be non-negative, got {ply}")
        _require(result in (0.0, 0.5, 1.0), f"invalid game result: {result}")

        placement = fields[0]
        _require(
            placement.count("K") == 1 and placement.count("k") == 1,
            f"invalid king count in placement: {placement}",
        )

        row_feats = [float(x) for x in feature_matrix[idx]]
        _require(
            np.allclose(
                np.asarray(row_feats[:7], dtype=np.float32),
                kcp13.material_block(fen, side),
                rtol=0,
                atol=1e-6,
            ),
            f"material block mismatch in row {gid}:{ply}",
        )

        rows.append(
            {
                "id": f"{gid}:{ply}",
                "group_id": gid,
                "game_id": gid,
                "fen": fen,
                "side": side,
                "ply": ply,
                "result": result,
                "features": row_feats,
            }
        )

    # Exclude positions the sealed bundle may not repeat. The key comes from the benchmark itself,
    # so this tool and `prepare_final` cannot disagree about what counts as the same position, and
    # the source is admitted by the benchmark's own loader before a single row is dropped.
    exclusion: dict[str, Any] | None = None
    if cfg.exclude_positions_from is not None:
        source_dir = Path(cfg.exclude_positions_from).resolve()
        _require(source_dir.is_dir(), "exclusion source is not a dataset directory")
        try:
            excluded_manifest, excluded_rows = benchmark_core.load_dataset(source_dir)
        except Exception as e:
            raise DatasetError(f"exclusion source is not an admissible dataset: {e}") from e
        # Draw rows are audited too: a position is excluded because it was seen, not because it
        # was scored, and the draw exclusion is applied downstream by the benchmark.
        seen = {position_key(row) for row in excluded_rows}
        kept = [row for row in rows if position_key(row) not in seen]
        exclusion = {
            "source_sha256": benchmark_core.digest(excluded_manifest),
            "removed_rows": len(rows) - len(kept),
        }
        rows = kept
        _require(bool(rows), "every row was excluded; nothing left to package")

    # Check game-level split assignments consistency
    assignments(rows)

    # Determine license text
    if cfg.license_file is not None:
        lic_p = Path(cfg.license_file).resolve()
        _require(lic_p.is_file(), f"license file not found: {lic_p}")
        license_text = lic_p.read_text(encoding="utf-8")
    elif cfg.license_text is not None:
        license_text = cfg.license_text
    elif cfg.kind == "owner-controlled":
        license_text = DEFAULT_OWNER_LICENSE
    elif cfg.kind == "synthetic":
        license_text = DEFAULT_SYNTHETIC_LICENSE
    else:
        raise DatasetError(f"license text or license file is required for kind {cfg.kind!r}")

    # Compute source_sha256
    if cfg.source_sha256 is not None:
        source_sha256 = cfg.source_sha256
    elif input_paths:
        source_sha256 = _compute_source_digest(input_paths)
    else:
        source_sha256 = hashlib.sha256(b"in-memory-dataframe").hexdigest()

    # Stage files
    staging_dir = out_path.with_name(f".{out_path.name}.staging-{os.getpid()}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    try:
        # Write rows.json
        rows_path = staging_dir / ROWS_FILE
        with rows_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        rows_sha256 = kcp13.sha256_of(rows_path)

        # Write license.txt
        license_path = staging_dir / LICENSE_FILE
        license_path.write_text(license_text, encoding="utf-8")
        license_evidence_sha256 = kcp13.sha256_of(license_path)

        # Write manifest.json
        manifest = {
            "schema": "playground-rows-v1",
            "version": cfg.version,
            "perspective": "side-to-move",
            "target": "decisive-game-outcome",
            "kind": cfg.kind,
            "license": cfg.license_name,
            "license_evidence_sha256": license_evidence_sha256,
            "rows_sha256": rows_sha256,
            "source_sha256": source_sha256,
            "golden_sha256": golden_sha256,
            "feature_schema": cfg.feature_schema,
            "engine_version": engine_ver,
            "columns": feature_cols,
        }
        if exclusion is not None:
            # Only present when rows were actually withheld, so the dataset identity the seal binds
            # cannot be the same for a filtered and an unfiltered bundle — and an unfiltered export
            # keeps the manifest it has always written.
            manifest["excluded_positions_source_sha256"] = exclusion["source_sha256"]
            manifest["excluded_positions_rows"] = exclusion["removed_rows"]
        manifest_path = staging_dir / MANIFEST_FILE
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_sha256 = benchmark_core.digest(manifest)

        # Validate with the real benchmark consumer
        benchmark_core.load_dataset(staging_dir)

        # Publish atomically
        staging_dir.rename(out_path)

        return {
            "schema": "playground-rows-v1",
            "version": cfg.version,
            "kind": cfg.kind,
            "engine_version": engine_ver,
            "rows": len(rows),
            "games": len(set(r["game_id"] for r in rows)),
            "manifest_sha256": manifest_sha256,
            "rows_sha256": rows_sha256,
            "golden_sha256": golden_sha256,
            "source_sha256": source_sha256,
            "license_evidence_sha256": license_evidence_sha256,
            "excluded_positions": exclusion,
        }
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
