"""Predeclared offline feature schema ablation runner (Issue #17).

Evaluates S0 (kcp-13), S1 (kcp-mobility-27-v1), and S2 (kcp-mobility-pawns-31-v1)
under the frozen protocol in docs/ablation/protocol-v1.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dicechess_training.benchmark.metrics import confidence, scores
from dicechess_training.contracts import (
    SCHEMA_CONTRACTS,
    kcp13,
)
from dicechess_training.schema import read_enriched_shards

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_PATH = ROOT / "docs/ablation/protocol-v1.json"


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _safe_protocol_path(p: Path, base: Path = ROOT) -> Path:
    resolved = p.resolve()
    base_resolved = base.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(f"Path traversal detected: {p}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Protocol file not found: {resolved}")
    return resolved


class ValueMLP(nn.Module):
    """Candidate evaluation model: identical MLP capacity across all schemas."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    config: dict,
    seed: int,
) -> ValueMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ValueMLP(input_dim, config["hidden_dims"])
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.0,
    )
    criterion = nn.BCELoss()

    raw_epochs = config.get("epochs", 5)
    epochs = int(raw_epochs) if isinstance(raw_epochs, (int, str)) else 5
    if not 1 <= epochs <= 100:
        raise ValueError(f"epochs must be in [1, 100], got {epochs}")

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def predict(model: ValueMLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        out = model(t).squeeze(1).numpy()
    return np.clip(out.astype(np.float64), 0.0, 1.0)


def compute_slices(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict[str, dict]:
    """Compute score slices matching the benchmark specification."""
    f = df[list(kcp13.COLUMN_NAMES)].to_numpy(dtype=np.float32)
    ply = df["ply"].to_numpy()
    side = df["side"].to_numpy()

    total_mat = f[:, 6]
    mat_diff = f[:, 5]

    phase_labels = np.full(len(df), "middlegame", dtype=object)
    phase_labels[total_mat <= 20] = "endgame"
    phase_labels[ply <= 10] = "opening"

    mat_labels = np.full(len(df), "balanced", dtype=object)
    mat_labels[mat_diff < -1] = "behind"
    mat_labels[mat_diff > 1] = "ahead"

    slice_definitions = {
        "phase:opening": phase_labels == "opening",
        "phase:middlegame": phase_labels == "middlegame",
        "phase:endgame": phase_labels == "endgame",
        "side:w": side == "w",
        "side:b": side == "b",
        "material:behind": mat_labels == "behind",
        "material:balanced": mat_labels == "balanced",
        "material:ahead": mat_labels == "ahead",
        "tactical:king_attack": f[:, 9] > 0,
        "tactical:king_danger": f[:, 10] > 0,
        "tactical:queen_attack": f[:, 11] > 0,
        "tactical:queen_danger": f[:, 12] > 0,
    }

    result = {}
    for name, mask in slice_definitions.items():
        if mask.sum() > 0:
            result[name] = scores(y[mask], p[mask])
    return result


def evaluate_probe_predictions(
    preds_by_probe_id: dict[str, float],
    protocol: dict,
) -> dict[str, Any]:
    tol = protocol["probes"]["equal_tolerance"]
    checks = {}
    for name in sorted(preds_by_probe_id):
        if name.endswith("-twin"):
            base_name = name.removesuffix("-twin")
            if base_name in preds_by_probe_id:
                checks[f"twin:{name}"] = (
                    abs(preds_by_probe_id[name] - preds_by_probe_id[base_name]) <= tol
                )

    checks["opening:start-w==start-w-6field"] = (
        abs(preds_by_probe_id["start-w"] - preds_by_probe_id["start-w-6field"]) <= tol
    )
    checks["opening:start-w==start-b"] = (
        abs(preds_by_probe_id["start-w"] - preds_by_probe_id["start-b"]) <= tol
    )
    checks["opening:start-w~0.5"] = (
        abs(preds_by_probe_id["start-w"] - 0.5) <= protocol["probes"]["opening_distance_from_half"]
    )
    checks["canonical:ep-e6==ep-none"] = (
        abs(preds_by_probe_id["ep-e6-w"] - preds_by_probe_id["ep-none-w"]) <= tol
    )
    checks["material:knight-up>start-w>knight-down"] = (
        preds_by_probe_id["knight-up-w"]
        > preds_by_probe_id["start-w"]
        > preds_by_probe_id["knight-down-b"]
    )
    checks["endgame:kings-only-finite"] = bool(np.isfinite(preds_by_probe_id["kings-only-w"]))

    return {
        "checks": {k: bool(v) for k, v in checks.items()},
        "predictions": {k: round(float(v), 5) for k, v in preds_by_probe_id.items()},
    }


def predict_probes(model: ValueMLP, schema_id: str, protocol: dict) -> dict[str, float]:
    contract = SCHEMA_CONTRACTS[schema_id]
    engine_ver = protocol.get("engine_version")
    corpus = (
        contract.load_golden(contract.golden_path(engine_ver))
        if engine_ver
        else contract.load_golden()
    )
    authored_probes = [p for p in corpus.probes if not p.id.startswith("sample-")]
    features = np.stack([p.features for p in authored_probes])
    preds = predict(model, features)
    return dict(zip((p.id for p in authored_probes), (float(x) for x in preds), strict=True))


def evaluate_probe_suite(model: ValueMLP, schema_id: str, protocol: dict) -> dict[str, Any]:
    preds = predict_probes(model, schema_id, protocol)
    return evaluate_probe_predictions(preds, protocol)


def _prepare_dataset_splits(base_df: pd.DataFrame, protocol: dict):
    game_hashes = {
        gid: int(hashlib.sha256(("playground-v1:" + str(gid)).encode()).hexdigest(), 16) % 10000
        for gid in base_df["game_id"].unique()
    }
    split_col = base_df["game_id"].map(game_hashes)
    decisive_mask = base_df["result"].isin([0.0, 1.0])

    train_mask = decisive_mask & (split_col < protocol["split"]["train_cutoff"])
    val_mask = decisive_mask & (split_col >= protocol["split"]["train_cutoff"])

    val_df = base_df[val_mask].reset_index(drop=True)
    y_val = val_df["result"].to_numpy(dtype=float)
    groups_val = val_df["game_id"].to_numpy()

    split_summary = {
        "total_positions": int(len(base_df)),
        "decisive_positions": int(decisive_mask.sum()),
        "train_positions": int(train_mask.sum()),
        "val_positions": int(val_mask.sum()),
        "train_games": int(len(set(base_df.loc[train_mask, "game_id"]))),
        "val_games": int(len(set(base_df.loc[val_mask, "game_id"]))),
    }
    return train_mask, val_mask, val_df, y_val, groups_val, split_summary


def _train_single_schema(
    s_key: str,
    sid: str,
    df_s: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    protocol: dict,
) -> tuple[dict[str, Any], np.ndarray]:
    contract = SCHEMA_CONTRACTS[sid]
    feature_cols = list(contract.COLUMN_NAMES)
    model_cfg = protocol["model"]
    seeds = protocol["seeds"]

    x_train = df_s.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
    y_train = df_s.loc[train_mask, "result"].to_numpy(dtype=np.float32)
    x_val = df_s.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)

    seed_results = []
    val_preds_all_seeds = []
    probe_preds_all_seeds: list[dict[str, float]] = []

    print(f"\nTraining and evaluating {s_key} ({sid}, {len(feature_cols)} features)...")
    for seed in seeds:
        model = train_model(x_train, y_train, len(feature_cols), model_cfg, seed)
        p_val = predict(model, x_val)
        val_preds_all_seeds.append(p_val)

        p_probes = predict_probes(model, sid, protocol)
        probe_preds_all_seeds.append(p_probes)

        sc = scores(y_val, p_val)
        sl = compute_slices(val_df, y_val, p_val)
        pr = evaluate_probe_predictions(p_probes, protocol)
        seed_results.append({"scores": sc, "slices": sl, "probes": pr})
        print(
            f"  seed {seed:3d}: log_loss = {sc['log_loss']:.4f}, "
            f"brier = {sc['brier']:.4f}, ece = {sc['ece']:.4f}"
        )

    all_probe_ids = list(probe_preds_all_seeds[0].keys())
    mean_probe_preds = {
        pid: float(np.mean([seed_p[pid] for seed_p in probe_preds_all_seeds]))
        for pid in all_probe_ids
    }
    probe_suite_mean = evaluate_probe_predictions(mean_probe_preds, protocol)

    p_val_mean = np.mean(val_preds_all_seeds, axis=0)
    schema_res = {
        "schema_id": sid,
        "feature_count": len(feature_cols),
        "mean_scores": scores(y_val, p_val_mean),
        "mean_slices": compute_slices(val_df, y_val, p_val_mean),
        "seed_evaluations": seed_results,
        "probe_suite_mean": probe_suite_mean,
    }
    return schema_res, p_val_mean


def _evaluate_gate_for_candidate(
    s_key: str,
    s_scores: dict[str, Any],
    p_val_s: np.ndarray,
    p_val_s0: np.ndarray,
    s0_ll: float,
    s0_brier: float,
    s0_ece: float,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    results_by_schema: dict[str, Any],
    protocol: dict,
) -> dict[str, Any]:
    ci = confidence(
        y_val,
        p_val_s,
        groups_val,
        reference=p_val_s0,
        repeats=protocol["uncertainty"]["repeats"],
        seed=protocol["uncertainty"]["seed"],
    )

    rel_log_loss_gain = (s0_ll - s_scores["log_loss"]) / s0_ll
    brier_regression = s_scores["brier"] - s0_brier
    ece_regression = s_scores["ece"] - s0_ece
    ll_ci_upper = ci["intervals"]["log_loss_delta"][1]

    s0_slices = results_by_schema["S0"]["mean_slices"]
    s_slices = results_by_schema[s_key]["mean_slices"]
    slice_regressions = {}
    max_slice_ll_reg = 0.0
    max_slice_brier_reg = 0.0

    for sname in s0_slices:
        if sname in s_slices:
            ll_diff = s_slices[sname]["log_loss"] - s0_slices[sname]["log_loss"]
            brier_diff = s_slices[sname]["brier"] - s0_slices[sname]["brier"]
            slice_regressions[sname] = {
                "log_loss_delta": ll_diff,
                "brier_delta": brier_diff,
            }
            max_slice_ll_reg = max(max_slice_ll_reg, ll_diff)
            max_slice_brier_reg = max(max_slice_brier_reg, brier_diff)

    gate_rules = protocol["gate"]
    passed_rel_gain = rel_log_loss_gain >= gate_rules["min_relative_log_loss_gain"]
    passed_ci_upper = ll_ci_upper < 0
    passed_brier = brier_regression <= gate_rules["max_brier_regression"]
    passed_ece = ece_regression <= gate_rules["max_ece_regression"]
    passed_slice_ll = max_slice_ll_reg <= gate_rules["max_slice_log_loss_regression"]
    passed_slice_brier = max_slice_brier_reg <= gate_rules["max_slice_brier_regression"]

    cleared = (
        passed_rel_gain
        and passed_ci_upper
        and passed_brier
        and passed_ece
        and passed_slice_ll
        and passed_slice_brier
    )

    return {
        "cleared": cleared,
        "relative_log_loss_gain": rel_log_loss_gain,
        "log_loss_delta_ci_95": ci["intervals"]["log_loss_delta"],
        "brier_regression": brier_regression,
        "ece_regression": ece_regression,
        "max_slice_log_loss_regression": max_slice_ll_reg,
        "max_slice_brier_regression": max_slice_brier_reg,
        "confidence": ci,
        "slice_regressions": slice_regressions,
        "checks": {
            "relative_log_loss_gain_gte_1pct": passed_rel_gain,
            "paired_ci_upper_lt_0": passed_ci_upper,
            "brier_no_regression": passed_brier,
            "ece_regression_lte_0_01": passed_ece,
            "slice_log_loss_lte_0_01": passed_slice_ll,
            "slice_brier_lte_0_01": passed_slice_brier,
        },
    }


def _select_schema(gate_evaluations: dict[str, Any], results_by_schema: dict[str, Any]) -> str:
    if gate_evaluations["S2"]["cleared"] and gate_evaluations["S1"]["cleared"]:
        s2_ll = results_by_schema["S2"]["mean_scores"]["log_loss"]
        s1_ll = results_by_schema["S1"]["mean_scores"]["log_loss"]
        return "S2" if s2_ll < s1_ll else "S1"
    if gate_evaluations["S2"]["cleared"]:
        return "S2"
    if gate_evaluations["S1"]["cleared"]:
        return "S1"
    return "S0"


def _validate_schema_row_alignment(schema_dfs: dict[str, pd.DataFrame]) -> None:
    """Ensure all schema DataFrames are non-empty and strictly aligned by row."""
    if not schema_dfs:
        raise ValueError("No schema DataFrames provided")
    base_key = "S0" if "S0" in schema_dfs else next(iter(schema_dfs))
    base_df = schema_dfs[base_key]
    base_game_ids = base_df["game_id"].to_numpy()
    base_plies = base_df["ply"].to_numpy()
    base_len = len(base_df)

    for key, df in schema_dfs.items():
        if key == base_key:
            continue
        if len(df) != base_len:
            raise ValueError(
                f"Schema {key} row count ({len(df)}) does not match {base_key} ({base_len})"
            )
        if not np.array_equal(df["game_id"].to_numpy(), base_game_ids):
            raise ValueError(f"Schema {key} game_id sequence does not match {base_key}")
        if not np.array_equal(df["ply"].to_numpy(), base_plies):
            raise ValueError(f"Schema {key} ply sequence does not match {base_key}")


def run_ablation(
    protocol_path: Path | None = None,
    enriched_base_dir: Path | None = None,
) -> dict:
    if protocol_path is None:
        protocol_path = DEFAULT_PROTOCOL_PATH
    if enriched_base_dir is None:
        enriched_base_dir = ROOT / "data/enriched"

    safe_protocol = _safe_protocol_path(protocol_path)
    protocol_bytes = safe_protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = sha256_of_bytes(protocol_bytes)
    print(f"Loaded protocol: {protocol['protocol_version']} (SHA-256: {protocol_sha256[:12]}...)")

    schema_dfs = {}
    for schema_key, schema_info in protocol["schemas"].items():
        sid = schema_info["schema_id"]
        shard_dir = enriched_base_dir / sid
        print(f"Loading enriched shards for {schema_key} ({sid}) from {shard_dir}...")
        df = read_enriched_shards(shard_dir, sid, protocol["engine_version"])
        schema_dfs[schema_key] = df

    _validate_schema_row_alignment(schema_dfs)

    base_df = schema_dfs["S0"]
    train_mask, val_mask, val_df, y_val, groups_val, split_summary = _prepare_dataset_splits(
        base_df, protocol
    )

    results_by_schema: dict[str, dict] = {}
    val_preds_by_schema: dict[str, np.ndarray] = {}

    for s_key, s_info in protocol["schemas"].items():
        res, preds = _train_single_schema(
            s_key,
            s_info["schema_id"],
            schema_dfs[s_key],
            train_mask,
            val_mask,
            val_df,
            y_val,
            protocol,
        )
        results_by_schema[s_key] = res
        val_preds_by_schema[s_key] = preds

    s0_ll = results_by_schema["S0"]["mean_scores"]["log_loss"]
    s0_brier = results_by_schema["S0"]["mean_scores"]["brier"]
    s0_ece = results_by_schema["S0"]["mean_scores"]["ece"]
    p_val_s0 = val_preds_by_schema["S0"]

    gate_evaluations = {}
    for s_key in ["S1", "S2"]:
        gate_evaluations[s_key] = _evaluate_gate_for_candidate(
            s_key,
            results_by_schema[s_key]["mean_scores"],
            val_preds_by_schema[s_key],
            p_val_s0,
            s0_ll,
            s0_brier,
            s0_ece,
            y_val,
            groups_val,
            results_by_schema,
            protocol,
        )

    selected_schema = _select_schema(gate_evaluations, results_by_schema)
    decision_record = {
        "selected_schema": selected_schema,
        "selected_schema_id": protocol["schemas"][selected_schema]["schema_id"],
        "gate_results": {k: v["cleared"] for k, v in gate_evaluations.items()},
    }

    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "engine_version": protocol["engine_version"],
        "split_summary": split_summary,
        "schemas": results_by_schema,
        "gate_evaluations": gate_evaluations,
        "decision": decision_record,
    }
