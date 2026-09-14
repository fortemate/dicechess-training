"""Run the JVM producer and Python reader against synthetic golden positions (requires sbt)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dicechess_training.contracts import SCHEMA_CONTRACTS, kcp13
from dicechess_training.schema import read_enriched_shards, write_shard

ROOT = Path(__file__).resolve().parents[1]
JVM = ROOT / "tools/kcp13-golden"
ENGINE = "0.9.3"
MAIN = "dicechess.training.golden.EnrichShardsApp"


def run_producer(commands: list[str], *, expected_error: str | None = None) -> None:
    result = subprocess.run(
        ["sbt", "-batch", f"-Dengine.version={ENGINE}", *commands],
        cwd=JVM,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="", flush=True)
    if expected_error is not None:
        if result.returncode == 0 or expected_error not in result.stdout:
            raise RuntimeError(f"Expected producer rejection: {expected_error}")
    elif result.returncode != 0:
        raise RuntimeError(f"Unexpected producer exit code {result.returncode}")


def main() -> None:
    corpus = kcp13.load_golden(kcp13.golden_path(ENGINE))
    rows = pd.DataFrame(
        [
            {
                "game_id": probe.id,
                "ply": 0,
                "fen": probe.fen,
                "dice": "PPP",
                "side": probe.side,
                "result": 1.0,
            }
            for probe in corpus.probes
        ]
    )
    with tempfile.TemporaryDirectory(prefix="ablation-enrichment-") as directory:
        base = Path(directory)
        raw = base / "raw"
        raw.mkdir()
        write_shard(rows, str(raw / "synthetic.parquet"))
        commands = [f'runMain {MAIN} "{raw}" "{base / sid}" {sid}' for sid in SCHEMA_CONTRACTS]
        run_producer(["test", *commands])
        for sid, contract in SCHEMA_CONTRACTS.items():
            actual = read_enriched_shards(base / sid, sid, ENGINE).set_index("game_id")
            golden = contract.load_golden(contract.golden_path(ENGINE))
            assert len(actual) == len(golden.probes)
            for probe in golden.probes:
                np.testing.assert_allclose(
                    actual.loc[probe.id, list(contract.COLUMN_NAMES)].to_numpy(dtype=np.float32),
                    probe.features,
                    rtol=1e-6,
                    atol=1e-6,
                )
            print(f"PASS: {sid} JVM metadata, row count, and all golden features", flush=True)

        # A valid side code contradicting the FEN must not be stamped as mover-canonical.
        bad = base / "bad"
        bad.mkdir()
        mismatched = rows.iloc[:1].copy()
        mismatched.loc[:, "side"] = "b"
        write_shard(mismatched, str(bad / "mismatch.parquet"))
        output = base / "rejected"
        run_producer(
            [f'runMain {MAIN} "{bad}" "{output}" kcp-13'], expected_error="side/FEN mismatch"
        )
        assert not list(output.glob("*.parquet")), "Failed enrichment published a shard"
        assert not list(output.glob("*.tmp")), "Failed enrichment left a partial shard"
        print("PASS: side/FEN mismatch rejected without publishing partial data", flush=True)


if __name__ == "__main__":
    main()
