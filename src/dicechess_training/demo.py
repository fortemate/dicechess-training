"""End-to-end demo: data -> game-level split -> train -> metrics -> ONNX -> parity.

Runs on CPU in well under a minute. By default it trains on the committed
bot-vs-bot sample (`sample/playsite-bots-v0`, see `sample/README.md`); with
`--synthetic` it falls back to random placements, which exercise the same
plumbing while carrying no signal at all.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from . import data, features, onnx_io, schema, train

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample" / "playsite-bots-v0"


def _load_sample(sample_dir: Path) -> tuple[str, pd.DataFrame]:
    print(f"[1/6] Loading the bot-vs-bot sample from {sample_dir}...")
    return "sample", schema.read_shards(str(sample_dir))


def _generate_synthetic(n_games: int, seed: int) -> tuple[str, pd.DataFrame]:
    print(f"[1/6] Generating {n_games} synthetic toy games (no signal by construction)...")
    df = data.generate_toy_games(n_games, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = str(Path(tmp) / "toy-000.parquet")
        schema.write_shard(df, shard_path)
        return "synthetic", schema.read_shard(shard_path)


# The sample holds ~40k positions, which the full-size 774-256-256-1 net memorizes within
# two epochs (holdout log-loss lands ABOVE the no-information baseline). A 32-unit net for two
# epochs generalizes instead, beating the baseline by ~0.04-0.05 nats across seeds. The lesson
# is the program's whole premise: at this data scale the labels bind, not the architecture.
DEMO_HIDDEN = 32


def run(
    n_games: int = 400,
    epochs: int = 2,
    out_dir: str = "out",
    seed: int = 0,
    synthetic: bool = False,
    sample_dir: str | None = None,
    hidden: int = DEMO_HIDDEN,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved_sample = Path(sample_dir) if sample_dir else SAMPLE_DIR

    if synthetic or not resolved_sample.exists():
        source, df = _generate_synthetic(n_games, seed)
    else:
        source, df = _load_sample(resolved_sample)

    print(f"[2/6] {len(df)} positions across {df['game_id'].nunique()} games (schema v0 verified).")

    print("[3/6] Game-level split (no position leakage between train and holdout)...")
    train_df, val_df = data.split_by_game(df, val_fraction=0.2, seed=seed)
    print(f"      train: {len(train_df)} positions, holdout: {len(val_df)} positions")

    print("[4/6] Encoding features and training the value MLP on CPU...")
    train_x, train_y = features.encode_frame(train_df), train_df["result"].to_numpy()
    val_x, val_y = features.encode_frame(val_df), val_df["result"].to_numpy()
    model = train.train_value_model(train_x, train_y, epochs=epochs, seed=seed, hidden=hidden)

    print("[5/6] Holdout metrics (judge on log-loss and calibration, not accuracy):")
    metrics = train.evaluate(model, val_x, val_y)
    baseline = train.no_information_log_loss(val_y)
    verdict = "better" if metrics["log_loss"] < baseline else "WORSE"
    print(
        f"      log-loss: {metrics['log_loss']:.4f}   "
        f"Brier: {metrics['brier']:.4f}   n={metrics['n']}"
    )
    print(
        f"      no-information baseline (always predict the base rate): {baseline:.4f}"
        f" -> the model is {verdict}"
    )
    for row in metrics["calibration"]:
        print(
            f"      bin {row['bin']}: predicted {row['mean_predicted']:.3f}"
            f" vs outcome {row['mean_outcome']:.3f} (n={row['count']})"
        )
    if source == "synthetic":
        print(
            "      (synthetic data carries no signal: matching the baseline is the correct result)"
        )

    print("[6/6] Exporting ONNX and checking PyTorch <-> onnxruntime parity...")
    onnx_path = str(out / f"value-{source}.onnx")
    onnx_io.export_value_model(model, onnx_path)
    diff = onnx_io.onnx_parity(model, onnx_path)
    print(f"      wrote {onnx_path}; max parity diff = {diff:.2e}")
    if diff > 1e-5:
        raise SystemExit(f"ONNX parity check FAILED: {diff:.2e} > 1e-5")

    print("Done.")
    metrics["onnx_parity"] = diff
    metrics["onnx_path"] = onnx_path
    metrics["source"] = source
    metrics["baseline_log_loss"] = baseline
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400, help="synthetic games (--synthetic only)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--out", default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--synthetic", action="store_true", help="use toy data instead of the sample"
    )
    parser.add_argument("--sample-dir", default=None, help="directory of schema-v0 Parquet shards")
    parser.add_argument("--hidden", type=int, default=DEMO_HIDDEN, help="hidden units per layer")
    args = parser.parse_args()
    run(
        n_games=args.games,
        epochs=args.epochs,
        out_dir=args.out,
        seed=args.seed,
        synthetic=args.synthetic,
        sample_dir=args.sample_dir,
        hidden=args.hidden,
    )


if __name__ == "__main__":
    main()
