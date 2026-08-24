"""End-to-end toy demo: data -> shards -> game-level split -> train -> ONNX -> parity.

Runs on CPU in well under a minute. Uses SYNTHETIC placeholder data until the
real bot-vs-bot sample lands (issue #4); the point is to exercise every joint
of the pipeline, not to produce a strong model.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from . import data, features, onnx_io, schema, train


def run(n_games: int = 400, epochs: int = 3, out_dir: str = "out", seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Generating {n_games} synthetic toy games (placeholder until issue #4)...")
    df = data.generate_toy_games(n_games, seed=seed)

    print("[2/6] Writing and re-reading Parquet shards (schema v0 round-trip)...")
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = str(Path(tmp) / "toy-000.parquet")
        schema.write_shard(df, shard_path)
        df = schema.read_shard(shard_path)

    print("[3/6] Game-level split (no position leakage between train and holdout)...")
    train_df, val_df = data.split_by_game(df, val_fraction=0.2, seed=seed)
    print(f"      train: {len(train_df)} positions, holdout: {len(val_df)} positions")

    print("[4/6] Encoding features and training the value MLP on CPU...")
    train_x, train_y = features.encode_frame(train_df), train_df["result"].to_numpy()
    val_x, val_y = features.encode_frame(val_df), val_df["result"].to_numpy()
    model = train.train_value_model(train_x, train_y, epochs=epochs, seed=seed)

    print("[5/6] Holdout metrics (judge on log-loss and calibration, not accuracy):")
    metrics = train.evaluate(model, val_x, val_y)
    print(
        f"      log-loss: {metrics['log_loss']:.4f}   "
        f"Brier: {metrics['brier']:.4f}   n={metrics['n']}"
    )
    for row in metrics["calibration"]:
        print(
            f"      bin {row['bin']}: predicted {row['mean_predicted']:.3f}"
            f" vs outcome {row['mean_outcome']:.3f} (n={row['count']})"
        )

    print("[6/6] Exporting ONNX and checking PyTorch <-> onnxruntime parity...")
    onnx_path = str(out / "value-toy.onnx")
    onnx_io.export_value_model(model, onnx_path)
    diff = onnx_io.onnx_parity(model, onnx_path)
    print(f"      wrote {onnx_path}; max parity diff = {diff:.2e}")
    if diff > 1e-5:
        raise SystemExit(f"ONNX parity check FAILED: {diff:.2e} > 1e-5")

    print("Done. Every joint of the pipeline works; labels are toys until issue #4.")
    metrics["onnx_parity"] = diff
    metrics["onnx_path"] = onnx_path
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", default="out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(n_games=args.games, epochs=args.epochs, out_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
