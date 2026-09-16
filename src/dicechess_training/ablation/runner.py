"""Offline feature schema ablation runner (Issue #17).

Evaluates S0 (kcp-13), S1 (kcp-mobility-27-v1), and S2 (kcp-mobility-pawns-31-v1)
under protocol v3, with historical v1/v2 training behavior preserved.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dicechess_training.ablation.extraction_cost import validate_extraction_cost
from dicechess_training.ablation.paths import resolve_read_path
from dicechess_training.benchmark.metrics import confidence, losses, scores
from dicechess_training.benchmark.splits import leakage, position_key
from dicechess_training.contracts import (
    SCHEMA_CONTRACTS,
    kcp13,
)
from dicechess_training.schema import read_enriched_shards, shard_content_digest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_PATH = ROOT / "docs/ablation/protocol-v4.json"


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _safe_protocol_path(p: Path) -> Path:
    return resolve_read_path(p)


class ValueMLP(nn.Module):
    """Candidate evaluation model: identical MLP capacity across all schemas.

    The model consumes **raw** schema features. When the protocol asks for
    standardisation the training statistics are carried inside the model as persistent
    buffers and applied as a constant affine transform before the first layer, so the
    scaler travels with the checkpoint and with the exported graph and the serving
    contract of ADR 0001 keeps taking raw features (#27). Without those statistics no
    buffer is registered at all, which keeps the checkpoint layout of protocols v1-v3
    byte-identical.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        feature_mean: np.ndarray | None = None,
        feature_scale: np.ndarray | None = None,
    ):
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
        if (feature_mean is None) != (feature_scale is None):
            raise ValueError("feature_mean and feature_scale must be given together")
        self.standardised = feature_mean is not None
        if self.standardised:
            mean = np.asarray(feature_mean, dtype=np.float32).reshape(-1)
            scale = np.asarray(feature_scale, dtype=np.float32).reshape(-1)
            if mean.shape != (input_dim,) or scale.shape != (input_dim,):
                raise ValueError(
                    f"standardisation statistics must have shape ({input_dim},), "
                    f"got {mean.shape} and {scale.shape}"
                )
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
                raise ValueError("standardisation statistics must be finite")
            if np.any(scale <= 0.0):
                raise ValueError("standardisation scale must be positive")
            self.register_buffer("feature_mean", torch.tensor(mean))
            self.register_buffer("feature_scale", torch.tensor(scale))

    def _standardise(self, x: torch.Tensor) -> torch.Tensor:
        if not self.standardised:
            return x
        return (x - self.feature_mean) / self.feature_scale

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Unbounded training scores; keep the probability-facing checkpoint layout."""
        return self.net[:-1](self._standardise(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._standardise(x))


def standardisation_statistics(
    x_train: np.ndarray, mode: str | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Training-partition mean and scale for `mode`, or (None, None) when disabled.

    Constant columns keep a scale of 1.0 so a feature that never varies in the training
    partition is centred rather than amplified into noise.
    """
    if mode in (None, "none"):
        return None, None
    if mode != "train-statistics":
        raise ValueError(f"unsupported feature standardisation: {mode!r}")
    if x_train.ndim != 2 or len(x_train) == 0:
        raise ValueError("standardisation needs a non-empty 2-D training matrix")
    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.std(axis=0, dtype=np.float64)
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _resolve_epochs(config: dict) -> int:
    raw_epochs = config.get("epochs", 5)
    epochs = int(raw_epochs) if isinstance(raw_epochs, (int, str)) else 5
    if not 1 <= epochs <= 100:
        raise ValueError(f"epochs must be in [1, 100], got {epochs}")
    return epochs


def _fit_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    config: dict,
    seed: int,
    epochs: int,
    checkpoints: frozenset[int] | None = None,
):
    """Train for `epochs` and yield `(epoch, model)` after each epoch in `checkpoints`.

    One pass serves both plain training and budget selection, so a candidate budget never
    costs an independent training run and the sequence of updates is identical either way.
    """
    loss_name = config.get("loss", "bce")
    if loss_name not in ("bce", "bce-with-logits"):
        raise ValueError(f"unsupported training loss: {loss_name!r}")
    use_logits = loss_name == "bce-with-logits"

    torch.manual_seed(seed)
    np.random.seed(seed)

    mean, scale = standardisation_statistics(x_train, config.get("feature_standardisation"))
    model = ValueMLP(input_dim, config["hidden_dims"], mean, scale)
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
    criterion = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    wanted = frozenset({epochs}) if checkpoints is None else checkpoints
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model.forward_logits(batch_x) if use_logits else model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        if epoch in wanted:
            model.eval()
            yield epoch, model
            model.train()
    model.eval()


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    config: dict,
    seed: int,
    epochs: int | None = None,
) -> ValueMLP:
    budget = _resolve_epochs(config) if epochs is None else epochs
    if not 1 <= budget <= 100:
        raise ValueError(f"epochs must be in [1, 100], got {budget}")
    trained = None
    for _, model in _fit_epochs(x_train, y_train, input_dim, config, seed, budget):
        trained = model
    if trained is None:  # pragma: no cover - budget is validated above
        raise ValueError("training produced no model")
    trained.eval()
    return trained


def select_epoch_budget(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_inner: np.ndarray,
    y_inner: np.ndarray,
    input_dim: int,
    config: dict,
    seed: int,
) -> dict[str, Any]:
    """Pick the epoch count by log loss on an inner tuning split carved out of train.

    The outer validation partition is never touched here: it stays a reporting and gating
    surface, so a schema cannot buy its budget with the numbers it is judged on (#27).
    """
    selection = config["epoch_selection"]
    candidates = sorted({int(c) for c in selection["candidates"]})
    if not candidates or candidates[0] < 1 or candidates[-1] > 100:
        raise ValueError(f"epoch_selection.candidates must lie in [1, 100], got {candidates}")
    if len(x_inner) == 0:
        raise ValueError("epoch selection needs a non-empty inner tuning split")

    scored = []
    for epoch, model in _fit_epochs(
        x_fit, y_fit, input_dim, config, seed, candidates[-1], frozenset(candidates)
    ):
        scored.append((epoch, float(losses(y_inner, predict(model, x_inner))[0].mean())))
    best_epoch, best_loss = min(scored, key=lambda item: (item[1], item[0]))
    return {
        "selected_epochs": int(best_epoch),
        "inner_log_loss": best_loss,
        "candidates": [{"epochs": e, "inner_log_loss": ll} for e, ll in scored],
    }


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

    train_cutoff = protocol["split"]["train_cutoff"]
    val_cutoff = protocol["split"].get("val_cutoff", 9000)

    train_mask = decisive_mask & (split_col < train_cutoff)
    val_mask = decisive_mask & (split_col >= train_cutoff) & (split_col < val_cutoff)
    test_mask = decisive_mask & (split_col >= val_cutoff)

    # Inner tuning split for budget selection (#27): the last games of the train partition by
    # the same deterministic hash. It is carved out of train, never out of validation, so the
    # reporting and gating surface stays untouched by model selection.
    selection_cfg = protocol.get("model", {}).get("epoch_selection")
    inner_cutoff = selection_cfg.get("inner_cutoff") if selection_cfg else None
    if inner_cutoff is not None:
        if not 0 < inner_cutoff < train_cutoff:
            raise ValueError(
                f"epoch_selection.inner_cutoff must lie in (0, {train_cutoff}), got {inner_cutoff}"
            )
        inner_tuning_mask = (split_col[train_mask] >= inner_cutoff).to_numpy()
    else:
        inner_tuning_mask = np.zeros(int(train_mask.sum()), dtype=bool)

    val_df = base_df[val_mask].reset_index(drop=True)
    y_val = val_df["result"].to_numpy(dtype=float)
    groups_val = val_df["game_id"].to_numpy()

    # Position leakage audit across partitions
    split_tags = np.full(len(base_df), "excluded", dtype=object)
    split_tags[train_mask] = "train"
    split_tags[val_mask] = "validation"
    split_tags[test_mask] = "test"

    decisive_indices = np.nonzero(decisive_mask)[0]
    rows_for_leakage = [{"fen": base_df.at[idx, "fen"]} for idx in decisive_indices]
    splits_for_leakage = split_tags[decisive_indices].tolist()
    leakage_audit = leakage(rows_for_leakage, splits_for_leakage)

    # Unseen validation positions mask and count
    train_positions = {
        position_key(r)
        for r, s in zip(rows_for_leakage, splits_for_leakage, strict=True)
        if s == "train"
    }
    unseen_val_mask = np.array(
        [position_key({"fen": fen}) not in train_positions for fen in val_df["fen"]],
        dtype=bool,
    )
    unseen_val_count = int(unseen_val_mask.sum())

    split_summary = {
        "total_positions": int(len(base_df)),
        "decisive_positions": int(decisive_mask.sum()),
        "train_positions": int(train_mask.sum()),
        "val_positions": int(val_mask.sum()),
        "test_positions": int(test_mask.sum()),
        "train_games": int(len(set(base_df.loc[train_mask, "game_id"]))),
        "val_games": int(len(set(base_df.loc[val_mask, "game_id"]))),
        "test_games": int(len(set(base_df.loc[test_mask, "game_id"]))),
        "leakage": leakage_audit,
        "unseen_val_positions": unseen_val_count,
        "inner_tuning_positions": int(inner_tuning_mask.sum()),
        "inner_tuning_games": int(
            len(set(base_df.loc[train_mask, "game_id"].to_numpy()[inner_tuning_mask]))
        ),
    }
    return (
        train_mask,
        val_mask,
        val_df,
        y_val,
        groups_val,
        split_summary,
        unseen_val_mask,
        inner_tuning_mask,
    )


def _train_single_schema(
    s_key: str,
    sid: str,
    df_s: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    val_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    unseen_val_mask: np.ndarray,
    protocol: dict,
    inner_tuning_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray]]:
    contract = SCHEMA_CONTRACTS[sid]
    feature_cols = list(contract.COLUMN_NAMES)
    model_cfg = protocol["model"]
    seeds = protocol["seeds"]

    x_train = df_s.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
    x_val = df_s.loc[val_mask, feature_cols].to_numpy(dtype=np.float32)

    select_budget = bool(model_cfg.get("epoch_selection")) and inner_tuning_mask is not None
    if select_budget:
        if inner_tuning_mask.shape != (len(x_train),):
            raise ValueError("inner tuning mask must align with the training partition")
        x_fit, y_fit = x_train[~inner_tuning_mask], y_train[~inner_tuning_mask]
        x_inner, y_inner = x_train[inner_tuning_mask], y_train[inner_tuning_mask]

    seed_results = []
    val_preds_all_seeds = []
    probe_preds_all_seeds: list[dict[str, float]] = []
    budget_records: list[dict[str, Any]] = []

    has_unseen = bool(unseen_val_mask.sum() > 0)
    y_val_unseen = y_val[unseen_val_mask] if has_unseen else None

    print(f"\nTraining and evaluating {s_key} ({sid}, {len(feature_cols)} features)...")
    for seed in seeds:
        if select_budget:
            # Choose the budget on the inner split, then refit on the whole train partition so
            # the reported model still sees every training row.
            chosen = select_epoch_budget(
                x_fit, y_fit, x_inner, y_inner, len(feature_cols), model_cfg, seed
            )
            budget_records.append({"seed": seed, **chosen})
            epochs = chosen["selected_epochs"]
        else:
            epochs = None
        model = train_model(x_train, y_train, len(feature_cols), model_cfg, seed, epochs)
        p_val = predict(model, x_val)
        val_preds_all_seeds.append(p_val)

        p_probes = predict_probes(model, sid, protocol)
        probe_preds_all_seeds.append(p_probes)

        sc = scores(y_val, p_val)
        sl = compute_slices(val_df, y_val, p_val)
        pr = evaluate_probe_predictions(p_probes, protocol)
        sc_unseen = scores(y_val_unseen, p_val[unseen_val_mask]) if has_unseen else None
        seed_results.append(
            {
                "seed": seed,
                "scores": sc,
                "slices": sl,
                "probes": pr,
                "unseen_scores": sc_unseen,
            }
        )
        print(
            f"  seed {seed:3d}: log_loss = {sc['log_loss']:.4f}, "
            f"brier = {sc['brier']:.4f}, ece = {sc['ece']:.4f}"
        )

    # Primary estimand: single-model replication summary across independent seeds
    lls = [s["scores"]["log_loss"] for s in seed_results]
    brs = [s["scores"]["brier"] for s in seed_results]
    eces = [s["scores"]["ece"] for s in seed_results]
    single_model_summary = {
        "log_loss_mean": float(np.mean(lls)),
        "log_loss_std": float(np.std(lls)),
        "brier_mean": float(np.mean(brs)),
        "brier_std": float(np.std(brs)),
        "ece_mean": float(np.mean(eces)),
        "ece_std": float(np.std(eces)),
    }

    if has_unseen:
        u_lls = [s["unseen_scores"]["log_loss"] for s in seed_results if s["unseen_scores"]]
        u_brs = [s["unseen_scores"]["brier"] for s in seed_results if s["unseen_scores"]]
        u_eces = [s["unseen_scores"]["ece"] for s in seed_results if s["unseen_scores"]]
        single_model_unseen = {
            "count": int(unseen_val_mask.sum()),
            "log_loss_mean": float(np.mean(u_lls)),
            "log_loss_std": float(np.std(u_lls)),
            "brier_mean": float(np.mean(u_brs)),
            "brier_std": float(np.std(u_brs)),
            "ece_mean": float(np.mean(u_eces)),
            "ece_std": float(np.std(u_eces)),
        }
    else:
        single_model_unseen = {
            "count": 0,
            "status": "no-unseen-positions",
        }

    first_slices = seed_results[0]["slices"]
    single_model_slices = {}
    for sl_name in first_slices:
        sl_lls = [s["slices"][sl_name]["log_loss"] for s in seed_results if sl_name in s["slices"]]
        sl_brs = [s["slices"][sl_name]["brier"] for s in seed_results if sl_name in s["slices"]]
        single_model_slices[sl_name] = {
            "log_loss_mean": float(np.mean(sl_lls)),
            "log_loss_std": float(np.std(sl_lls)),
            "brier_mean": float(np.mean(sl_brs)),
            "brier_std": float(np.std(sl_brs)),
        }

    all_probe_ids = list(probe_preds_all_seeds[0].keys())
    mean_probe_preds = {
        pid: float(np.mean([seed_p[pid] for seed_p in probe_preds_all_seeds]))
        for pid in all_probe_ids
    }

    # Secondary diagnostic ensemble
    p_val_ensemble = np.mean(val_preds_all_seeds, axis=0)
    ensemble_diagnostic = {
        "scores": scores(y_val, p_val_ensemble),
        "slices": compute_slices(val_df, y_val, p_val_ensemble),
        "probes": evaluate_probe_predictions(mean_probe_preds, protocol),
    }

    schema_res = {
        "schema_id": sid,
        "feature_count": len(feature_cols),
        "training": {
            "feature_standardisation": model_cfg.get("feature_standardisation") or "none",
            "epoch_selection": (
                {
                    "method": model_cfg["epoch_selection"].get("method", "inner-tuning-split"),
                    "per_seed": budget_records,
                }
                if select_budget
                else {"method": "fixed", "epochs": _resolve_epochs(model_cfg)}
            ),
        },
        "single_model_summary": single_model_summary,
        "single_model_unseen": single_model_unseen,
        "single_model_slices": single_model_slices,
        "seed_evaluations": seed_results,
        "ensemble_diagnostic": ensemble_diagnostic,
    }
    return schema_res, p_val_ensemble, val_preds_all_seeds


def _paired_bootstrap_mean_estimand(
    y: np.ndarray,
    s_preds: list[np.ndarray],
    s0_preds: list[np.ndarray],
    groups: np.ndarray,
    repeats: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    if not isinstance(repeats, int) or not 1 <= repeats <= 10000:
        raise ValueError("bootstrap repeats must be an integer in [1, 10000]")
    bounded_repeats = max(1, min(int(repeats), 10000))

    names, inverse = np.unique(groups, return_inverse=True)
    if len(names) < 2:
        return {
            "method": "group-bootstrap-percentile-95-mean-seed-delta",
            "groups": len(names),
            "log_loss_delta_ci_95": None,
            "brier_delta_ci_95": None,
            "per_seed_log_loss_delta_cis": [],
            "reason": "fewer-than-two-groups",
        }

    per_seed_ll_deltas = []
    per_seed_br_deltas = []
    per_seed_cis = []
    for p_s, p_s0 in zip(s_preds, s0_preds, strict=True):
        ll_s, br_s = losses(y, p_s)
        ll_s0, br_s0 = losses(y, p_s0)
        per_seed_ll_deltas.append(ll_s - ll_s0)
        per_seed_br_deltas.append(br_s - br_s0)
        single_ci = confidence(y, p_s, groups, reference=p_s0, repeats=bounded_repeats, seed=seed)
        per_seed_cis.append(single_ci["intervals"]["log_loss_delta"])

    mean_row_ll_delta = np.mean(per_seed_ll_deltas, axis=0)
    mean_row_br_delta = np.mean(per_seed_br_deltas, axis=0)

    values = np.column_stack([np.ones(len(y)), mean_row_ll_delta, mean_row_br_delta])
    totals = np.zeros((len(names), 3))
    np.add.at(totals, inverse, values)

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bounded_repeats):
        sampled = totals[rng.integers(len(names), size=len(names))].sum(axis=0)
        total_n = sampled[0]
        samples.append([sampled[1] / total_n, sampled[2] / total_n])

    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "method": "group-bootstrap-percentile-95-mean-seed-delta",
        "groups": len(names),
        "log_loss_delta_ci_95": bounds[:, 0].tolist(),
        "brier_delta_ci_95": bounds[:, 1].tolist(),
        "per_seed_log_loss_delta_cis": per_seed_cis,
    }


def _evaluate_slice_regressions(
    s0_slices: dict[str, Any], s_slices: dict[str, Any]
) -> tuple[dict[str, Any], float, float]:
    slice_regressions = {}
    max_slice_ll_reg = 0.0
    max_slice_brier_reg = 0.0

    for sname in s0_slices:
        if sname in s_slices:
            ll_diff = s_slices[sname]["log_loss_mean"] - s0_slices[sname]["log_loss_mean"]
            brier_diff = s_slices[sname]["brier_mean"] - s0_slices[sname]["brier_mean"]
            slice_regressions[sname] = {
                "log_loss_delta": ll_diff,
                "brier_delta": brier_diff,
            }
            max_slice_ll_reg = max(max_slice_ll_reg, ll_diff)
            max_slice_brier_reg = max(max_slice_brier_reg, brier_diff)
    return slice_regressions, max_slice_ll_reg, max_slice_brier_reg


def _evaluate_unseen_gate(
    s_key: str,
    results_by_schema: dict[str, Any],
    val_preds_by_schema: dict[str, list[np.ndarray]],
    y_val: np.ndarray,
    groups_val: np.ndarray,
    unseen_val_mask: np.ndarray,
    protocol: dict,
) -> dict[str, Any]:
    unseen_count = int(unseen_val_mask.sum())
    gate_rules = protocol["gate"]
    if unseen_count == 0:
        return {
            "unseen_count": 0,
            "passed_exists": False,
            "passed_ll": False,
            "passed_brier": False,
            "passed": False,
            "reason": "no-unseen-positions",
            "log_loss_delta": None,
            "brier_delta": None,
            "log_loss_delta_ci_95": None,
        }

    s0_unseen = results_by_schema["S0"]["single_model_unseen"]
    s_unseen = results_by_schema[s_key]["single_model_unseen"]
    ll_delta = s_unseen["log_loss_mean"] - s0_unseen["log_loss_mean"]
    br_delta = s_unseen["brier_mean"] - s0_unseen["brier_mean"]

    max_ll_reg = gate_rules.get("max_unseen_log_loss_regression", 0.01)
    max_br_reg = gate_rules.get("max_unseen_brier_regression", 0.0)

    passed_ll = ll_delta <= max_ll_reg
    passed_br = br_delta <= max_br_reg

    raw_repeats = protocol["uncertainty"]["repeats"]
    repeats = int(raw_repeats) if isinstance(raw_repeats, (int, str)) else 1000
    if not 1 <= repeats <= 10000:
        raise ValueError(f"uncertainty.repeats must be in [1, 10000], got {repeats}")

    unseen_ci = _paired_bootstrap_mean_estimand(
        y_val[unseen_val_mask],
        [p[unseen_val_mask] for p in val_preds_by_schema[s_key]],
        [p[unseen_val_mask] for p in val_preds_by_schema["S0"]],
        groups_val[unseen_val_mask],
        repeats=repeats,
        seed=protocol["uncertainty"]["seed"],
    )

    passed_unseen = passed_ll and passed_br
    return {
        "unseen_count": unseen_count,
        "passed_exists": True,
        "passed_ll": passed_ll,
        "passed_brier": passed_br,
        "passed": passed_unseen,
        "reason": None if passed_unseen else "unseen-metrics-regressed",
        "log_loss_delta": ll_delta,
        "brier_delta": br_delta,
        "log_loss_delta_ci_95": unseen_ci["log_loss_delta_ci_95"],
    }


def _evaluate_extraction_cost_for_candidate(
    s_key: str,
    extraction_cost: dict[str, Any] | None,
    protocol: dict,
) -> dict[str, Any]:
    gate_rules = protocol["gate"]
    metric_key = gate_rules.get("extraction_cost_metric", "median_us")
    max_ov = gate_rules.get("max_extraction_latency_overhead", 0.10)

    if extraction_cost is None:
        return {
            "evidence_valid": False,
            "cost_cleared": False,
            "reason": "missing-extraction-cost-evidence",
            "mean_relative_overhead": None,
            "max_relative_overhead": None,
            "probe_overheads": {},
        }

    try:
        validate_extraction_cost(extraction_cost, protocol)
    except ValueError as exc:
        return {
            "evidence_valid": False,
            "cost_cleared": False,
            "reason": f"invalid-extraction-cost-evidence: {exc}",
            "mean_relative_overhead": None,
            "max_relative_overhead": None,
            "probe_overheads": {},
        }

    if gate_rules.get("extraction_cost_aggregation") != "mean_relative_overhead":
        raise ValueError("Unsupported extraction cost aggregation")
    if metric_key != "median_us":
        raise ValueError("Unsupported extraction cost metric")
    probe_overheads = {}
    for pid, pdata in extraction_cost["probes"].items():
        s0_val = pdata["schemas"]["S0"][metric_key]
        s_val = pdata["schemas"][s_key][metric_key]
        probe_overheads[pid] = float((s_val - s0_val) / s0_val)

    mean_overhead = math.fsum(probe_overheads.values()) / len(probe_overheads)
    max_overhead = float(np.max(list(probe_overheads.values())))
    cost_cleared = mean_overhead <= max_ov

    return {
        "evidence_valid": True,
        "cost_cleared": cost_cleared,
        "reason": None if cost_cleared else "extraction-latency-overhead-exceeded",
        "mean_relative_overhead": mean_overhead,
        "max_relative_overhead": max_overhead,
        "probe_overheads": probe_overheads,
    }


def no_information_reference(
    y_train: np.ndarray,
    y_val: np.ndarray,
    unseen_val_mask: np.ndarray,
    protocol: dict,
) -> dict[str, Any] | None:
    """Score the constant predictor the benchmark of #13 uses as its default reference.

    Absent from protocols v1-v3, which compared the schemas only against each other: a run in
    which every arm is worse than predicting the base rate still produced a confident winner.
    """
    admissibility = protocol.get("admissibility")
    if not admissibility:
        return None
    reference = admissibility.get("reference", "constant-train-base-rate")
    if reference != "constant-train-base-rate":
        raise ValueError(f"unsupported admissibility reference: {reference!r}")
    metric = admissibility.get("metric", "log_loss")
    if metric not in ("log_loss", "brier"):
        raise ValueError(f"unsupported admissibility metric: {metric!r}")

    base_rate = float(np.asarray(y_train, dtype=float).mean())
    prediction = np.full(len(y_val), base_rate, dtype=float)
    record = {
        "reference": reference,
        "metric": metric,
        "prediction": base_rate,
        "scores": scores(y_val, prediction),
    }
    if unseen_val_mask.sum() > 0:
        record["unseen_scores"] = scores(y_val[unseen_val_mask], prediction[unseen_val_mask])
    return record


def _admissibility_for_schema(
    summary: dict[str, Any], no_info: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Does this schema's mean single model beat the no-information reference?"""
    if no_info is None:
        return None
    metric = no_info["metric"]
    candidate = float(summary[f"{metric}_mean"])
    reference = float(no_info["scores"][metric])
    return {
        "metric": metric,
        "candidate": candidate,
        "reference": reference,
        "margin": reference - candidate,
        "admissible": bool(candidate < reference),
    }


def _evaluate_gate_for_candidate(
    s_key: str,
    results_by_schema: dict[str, Any],
    val_preds_by_schema: dict[str, list[np.ndarray]],
    y_val: np.ndarray,
    groups_val: np.ndarray,
    unseen_val_mask: np.ndarray,
    extraction_cost: dict[str, Any] | None,
    protocol: dict,
    no_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    s0_sm = results_by_schema["S0"]["single_model_summary"]
    s_sm = results_by_schema[s_key]["single_model_summary"]

    s0_ll = s0_sm["log_loss_mean"]
    s0_brier = s0_sm["brier_mean"]
    s0_ece = s0_sm["ece_mean"]

    rel_log_loss_gain = (s0_ll - s_sm["log_loss_mean"]) / s0_ll
    brier_regression = s_sm["brier_mean"] - s0_brier
    ece_regression = s_sm["ece_mean"] - s0_ece

    # Paired whole-game bootstrap on mean seed delta (primary estimand)
    raw_repeats = protocol["uncertainty"]["repeats"]
    repeats = int(raw_repeats) if isinstance(raw_repeats, (int, str)) else 1000
    if not 1 <= repeats <= 10000:
        raise ValueError(f"uncertainty.repeats must be in [1, 10000], got {repeats}")

    primary_ci = _paired_bootstrap_mean_estimand(
        y_val,
        val_preds_by_schema[s_key],
        val_preds_by_schema["S0"],
        groups_val,
        repeats=repeats,
        seed=protocol["uncertainty"]["seed"],
    )
    ll_ci = primary_ci["log_loss_delta_ci_95"]

    # Critical slices
    slice_regs, max_slice_ll_reg, max_slice_brier_reg = _evaluate_slice_regressions(
        results_by_schema["S0"]["single_model_slices"],
        results_by_schema[s_key]["single_model_slices"],
    )

    # Unseen positions evaluation
    unseen_eval = _evaluate_unseen_gate(
        s_key,
        results_by_schema,
        val_preds_by_schema,
        y_val,
        groups_val,
        unseen_val_mask,
        protocol,
    )

    # Extraction cost evaluation
    cost_eval = _evaluate_extraction_cost_for_candidate(s_key, extraction_cost, protocol)

    gate_rules = protocol["gate"]
    passed_rel_gain = bool(rel_log_loss_gain >= gate_rules["min_relative_log_loss_gain"])
    passed_ci_upper = bool((ll_ci is not None) and (ll_ci[1] < 0))
    passed_brier = bool(brier_regression <= gate_rules["max_brier_regression"])
    passed_ece = bool(ece_regression <= gate_rules["max_ece_regression"])
    passed_slice_ll = bool(max_slice_ll_reg <= gate_rules["max_slice_log_loss_regression"])
    passed_slice_brier = bool(max_slice_brier_reg <= gate_rules["max_slice_brier_regression"])
    passed_unseen = bool(unseen_eval["passed"])
    passed_cost = bool(cost_eval["cost_cleared"])

    # Admissibility floor (#27): a candidate that loses to the constant predictor cannot be
    # selected however favourable its relative comparison with S0 looks.
    admissibility = _admissibility_for_schema(s_sm, no_info)
    passed_admissible = True if admissibility is None else bool(admissibility["admissible"])

    cleared = (
        passed_rel_gain
        and passed_ci_upper
        and passed_brier
        and passed_ece
        and passed_slice_ll
        and passed_slice_brier
        and passed_unseen
        and passed_cost
        and passed_admissible
    )

    return {
        "cleared": cleared,
        "relative_log_loss_gain": rel_log_loss_gain,
        "log_loss_delta_ci_95": ll_ci,
        "brier_delta_ci_95": primary_ci["brier_delta_ci_95"],
        "per_seed_log_loss_delta_cis": primary_ci["per_seed_log_loss_delta_cis"],
        "brier_regression": brier_regression,
        "ece_regression": ece_regression,
        "max_slice_log_loss_regression": max_slice_ll_reg,
        "max_slice_brier_regression": max_slice_brier_reg,
        "slice_regressions": slice_regs,
        "unseen_evaluation": unseen_eval,
        "extraction_cost_evaluation": cost_eval,
        "admissibility": admissibility,
        "checks": {
            "relative_log_loss_gain_gte_1pct": passed_rel_gain,
            "paired_ci_upper_lt_0": passed_ci_upper,
            "brier_no_regression": passed_brier,
            "ece_regression_lte_0_01": passed_ece,
            "slice_log_loss_lte_0_01": passed_slice_ll,
            "slice_brier_lte_0_01": passed_slice_brier,
            "unseen_positions_exist": bool(unseen_eval["passed_exists"]),
            "unseen_log_loss_lte_0_01": bool(unseen_eval["passed_ll"]),
            "unseen_brier_no_regression": bool(unseen_eval["passed_brier"]),
            "extraction_cost_evidence_valid": bool(cost_eval["evidence_valid"]),
            "extraction_cost_lte_tolerance": passed_cost,
            **({} if admissibility is None else {"beats_no_information": passed_admissible}),
        },
    }


def _select_schema(
    gate_evaluations: dict[str, Any],
    results_by_schema: dict[str, Any],
    no_info: dict[str, Any] | None = None,
) -> str | None:
    """Selected schema, or `None` when even the baseline fails the admissibility floor.

    The relative gate decides between the candidates exactly as protocol v2 froze it. The floor
    only removes arms that are worse than predicting the base rate, the baseline included: a run
    in which S0 is inadmissible has measured nothing and must not hand a schema to training.
    """
    if gate_evaluations["S2"]["cleared"] and gate_evaluations["S1"]["cleared"]:
        s2_ll = results_by_schema["S2"]["single_model_summary"]["log_loss_mean"]
        s1_ll = results_by_schema["S1"]["single_model_summary"]["log_loss_mean"]
        return "S2" if s2_ll < s1_ll else "S1"
    if gate_evaluations["S2"]["cleared"]:
        return "S2"
    if gate_evaluations["S1"]["cleared"]:
        return "S1"
    s0_admissibility = _admissibility_for_schema(
        results_by_schema["S0"]["single_model_summary"], no_info
    )
    if s0_admissibility is not None and not s0_admissibility["admissible"]:
        return None
    return "S0"


def _validate_schema_keys(
    schema_dfs: dict[str, pd.DataFrame], base_key: str, base_len: int
) -> None:
    """Check that row counts match and (game_id, ply) keys are unique."""
    for key, df in schema_dfs.items():
        if len(df) != base_len:
            raise ValueError(
                f"Schema {key} row count ({len(df)}) does not match {base_key} ({base_len})"
            )
        keys = list(zip(df["game_id"], df["ply"], strict=True))
        if len(keys) != len(set(keys)):
            raise ValueError(f"Schema {key} contains duplicate (game_id, ply) row keys")


def _validate_schema_source_columns(
    schema_dfs: dict[str, pd.DataFrame], base_key: str, base_df: pd.DataFrame
) -> None:
    """Check that all candidate schemas contain identical source column values."""
    source_fields = ["game_id", "ply", "fen", "side", "dice", "result"]
    for key, df in schema_dfs.items():
        if key == base_key:
            continue
        for col in source_fields:
            if col not in df.columns:
                raise ValueError(f"Schema {key} missing source column {col!r}")
            if not (df[col].to_numpy() == base_df[col].to_numpy()).all():
                raise ValueError(f"Schema {key} source column {col!r} does not match {base_key}")


def _validate_schema_feature_prefixes(schema_dfs: dict[str, pd.DataFrame]) -> None:
    """Check that float32 feature vectors satisfy prefix identity (S0 in S1, S0/S1 in S2)."""
    if "S0" in schema_dfs:
        s0_cols = list(SCHEMA_CONTRACTS["kcp-13"].COLUMN_NAMES)
        f0 = schema_dfs["S0"][s0_cols].to_numpy(dtype=np.float32)
        if "S1" in schema_dfs:
            f1_prefix = schema_dfs["S1"][s0_cols].to_numpy(dtype=np.float32)
            if not np.array_equal(f0, f1_prefix):
                raise ValueError("Schema S1 float32 prefix does not match S0")
        if "S2" in schema_dfs:
            f2_prefix13 = schema_dfs["S2"][s0_cols].to_numpy(dtype=np.float32)
            if not np.array_equal(f0, f2_prefix13):
                raise ValueError("Schema S2 float32 13-feature prefix does not match S0")

    if "S1" in schema_dfs and "S2" in schema_dfs:
        s1_cols = list(SCHEMA_CONTRACTS["kcp-mobility-27-v1"].COLUMN_NAMES)
        f1 = schema_dfs["S1"][s1_cols].to_numpy(dtype=np.float32)
        f2_prefix27 = schema_dfs["S2"][s1_cols].to_numpy(dtype=np.float32)
        if not np.array_equal(f1, f2_prefix27):
            raise ValueError("Schema S2 float32 27-feature prefix does not match S1")


def _validate_cross_schema_integrity(schema_dfs: dict[str, pd.DataFrame]) -> None:
    """Ensure all schema DataFrames have unique keys, identical source rows, and prefix identity."""
    if not schema_dfs:
        raise ValueError("No schema DataFrames provided")
    base_key = "S0" if "S0" in schema_dfs else next(iter(schema_dfs))
    base_df = schema_dfs[base_key]
    base_len = len(base_df)

    _validate_schema_keys(schema_dfs, base_key, base_len)
    _validate_schema_source_columns(schema_dfs, base_key, base_df)
    _validate_schema_feature_prefixes(schema_dfs)


_validate_schema_row_alignment = _validate_cross_schema_integrity


def _load_extraction_cost(path: Path | None, protocol: dict) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_bytes())
    validate_extraction_cost(raw, protocol)
    return raw


def run_ablation(
    protocol_path: Path | None = None,
    enriched_base_dir: Path | None = None,
    extraction_cost_path: Path | None = None,
) -> dict:
    if protocol_path is None:
        protocol_path = DEFAULT_PROTOCOL_PATH
    if enriched_base_dir is None:
        enriched_base_dir = ROOT / "data/enriched"
    if extraction_cost_path is None:
        default_ext = ROOT / "tests/fixtures/benchmark/extraction-cost-0.9.3.json"
        if default_ext.exists():
            extraction_cost_path = default_ext

    safe_protocol = _safe_protocol_path(protocol_path)
    protocol_bytes = safe_protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = sha256_of_bytes(protocol_bytes)
    print(f"Loaded protocol: {protocol['protocol_version']} (SHA-256: {protocol_sha256[:12]}...)")

    extraction_cost = _load_extraction_cost(extraction_cost_path, protocol)

    schema_dfs = {}
    input_shard_digests = {}
    input_shard_content_digests = {}
    for schema_key, schema_info in protocol["schemas"].items():
        sid = schema_info["schema_id"]
        shard_dir = enriched_base_dir / sid
        print(f"Loading enriched shards for {schema_key} ({sid}) from {shard_dir}...")
        df = read_enriched_shards(shard_dir, sid, protocol["engine_version"])
        schema_dfs[schema_key] = df
        shard_files = sorted(shard_dir.glob("*.parquet"))
        input_shard_digests[schema_key] = {
            sf.name: sha256_of_bytes(sf.read_bytes()) for sf in shard_files
        }
        # File digests identify one producer run; content digests are what a re-run can match.
        input_shard_content_digests[schema_key] = {
            sf.name: shard_content_digest(sf) for sf in shard_files
        }

    _validate_cross_schema_integrity(schema_dfs)

    base_df = schema_dfs["S0"]
    (
        train_mask,
        val_mask,
        val_df,
        y_val,
        groups_val,
        split_summary,
        unseen_val_mask,
        inner_tuning_mask,
    ) = _prepare_dataset_splits(base_df, protocol)
    y_train = base_df.loc[train_mask, "result"].to_numpy(dtype=np.float32)

    results_by_schema: dict[str, dict] = {}
    val_preds_all_by_schema: dict[str, list[np.ndarray]] = {}

    for s_key, s_info in protocol["schemas"].items():
        res, _p_ens, seed_preds = _train_single_schema(
            s_key,
            s_info["schema_id"],
            schema_dfs[s_key],
            train_mask,
            val_mask,
            val_df,
            y_train,
            y_val,
            unseen_val_mask,
            protocol,
            inner_tuning_mask,
        )
        results_by_schema[s_key] = res
        val_preds_all_by_schema[s_key] = seed_preds

    no_info = no_information_reference(y_train, y_val, unseen_val_mask, protocol)
    if no_info is not None:
        for res in results_by_schema.values():
            res["admissibility"] = _admissibility_for_schema(res["single_model_summary"], no_info)
        print(
            f"\nNo-information reference ({no_info['reference']} = {no_info['prediction']:.4f}): "
            f"{no_info['metric']} = {no_info['scores'][no_info['metric']]:.4f}"
        )
        for s_key, res in results_by_schema.items():
            adm = res["admissibility"]
            verdict = "admissible" if adm["admissible"] else "INADMISSIBLE"
            print(
                f"  {s_key} ({res['schema_id']}): {adm['metric']} = {adm['candidate']:.4f} "
                f"({verdict}, margin {adm['margin']:+.4f})"
            )

    gate_evaluations = {}
    for s_key in ["S1", "S2"]:
        gate_evaluations[s_key] = _evaluate_gate_for_candidate(
            s_key,
            results_by_schema,
            val_preds_all_by_schema,
            y_val,
            groups_val,
            unseen_val_mask,
            extraction_cost,
            protocol,
            no_info,
        )

    selected_schema = _select_schema(gate_evaluations, results_by_schema, no_info)
    decision_record = {
        "status": (
            "provisional-development"
            if selected_schema is not None
            else "inadmissible-no-selection"
        ),
        "selected_schema": selected_schema,
        "selected_schema_id": (
            protocol["schemas"][selected_schema]["schema_id"]
            if selected_schema is not None
            else None
        ),
        "gate_results": {k: v["cleared"] for k, v in gate_evaluations.items()},
        "private_qualification_ref": (
            "Private Decision: Playground Feature Schema Qualification (Issue #17)"
        ),
    }

    if selected_schema is None:
        decision_record["inadmissible_reason"] = (
            "every schema, the baseline included, scored worse than the no-information "
            "reference on the primary metric; the run measured nothing selectable"
        )

    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "engine_version": protocol["engine_version"],
        "status": (
            "provisional-development"
            if selected_schema is not None
            else "inadmissible-no-selection"
        ),
        "split_summary": split_summary,
        "no_information_reference": no_info,
        "input_shard_digests": input_shard_digests,
        "input_shard_content_digests": input_shard_content_digests,
        "schemas": results_by_schema,
        "gate_evaluations": gate_evaluations,
        "extraction_cost": extraction_cost,
        "decision": decision_record,
    }
