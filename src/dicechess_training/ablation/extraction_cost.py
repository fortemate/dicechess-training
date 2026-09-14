"""Validate extraction measurements against the engine's complete golden probe corpus."""

from __future__ import annotations

import math
from typing import Any

from dicechess_training.contracts import kcp13

METRICS = ("min_us", "median_us", "p95_us", "max_us")


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _validate_header(raw: dict, engine_version: str) -> None:
    if raw.get("schema") != "playground-extraction-benchmark-v1":
        raise ValueError(f"Invalid extraction benchmark schema: {raw.get('schema')}")
    if raw.get("engine_version") != engine_version:
        raise ValueError(
            f"Extraction benchmark engine {raw.get('engine_version')} != protocol {engine_version}"
        )
    if raw.get("engine_artifact") != f"com.fortemate:dicechess-engine_3:{engine_version}":
        raise ValueError("Extraction benchmark engine artifact mismatch")
    runtime = _object(raw.get("runtime"), "runtime")
    for key in ("java_version", "java_vendor", "os_name", "os_arch"):
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            raise ValueError(f"Missing extraction runtime {key}")
    config = _object(raw.get("config"), "config")
    for key in ("warmup_iterations", "sample_iterations"):
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Extraction config {key} must be a positive integer")


def _validate_measurements(pdata: dict, pid: str, schemas: dict) -> None:
    measured = _object(pdata.get("schemas"), f"{pid}.schemas")
    if set(measured) != set(schemas):
        raise ValueError(f"Extraction schema coverage mismatch for {pid}")
    for key, schema in schemas.items():
        record = _object(measured[key], f"{pid}.{key}")
        if record.get("schema_id") != schema["schema_id"]:
            raise ValueError(f"Extraction schema identity mismatch for {pid}.{key}")
        values = [_positive_number(record.get(m), f"{pid}.{key}.{m}") for m in METRICS]
        if values != sorted(values):
            raise ValueError(f"Extraction latency quantiles out of order for {pid}.{key}")


def validate_extraction_cost(raw: Any, protocol: dict) -> None:
    """Reject partial, incompatible or invalid evidence before reporting or gate evaluation."""
    raw = _object(raw, "extraction benchmark")
    engine_version = protocol["engine_version"]
    _validate_header(raw, engine_version)
    corpus = kcp13.load_golden(kcp13.golden_path(engine_version))
    if corpus.engine_version != engine_version:
        raise ValueError("Extraction golden corpus engine mismatch")
    expected = corpus.by_id()
    probes = _object(raw.get("probes"), "extraction probes")
    if set(probes) != set(expected):
        raise ValueError("Extraction probe coverage must match the complete engine golden corpus")
    for pid, probe in expected.items():
        pdata = _object(probes[pid], f"probe {pid}")
        fen = pdata.get("fen")
        if not isinstance(fen, str) or fen.split() != probe.fen.split():
            raise ValueError(f"Extraction probe position mismatch for {pid}")
        _validate_measurements(pdata, pid, protocol["schemas"])
