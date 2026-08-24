import pytest

from dicechess_training import demo


def test_demo_on_synthetic_data(tmp_path):
    metrics = demo.run(n_games=30, epochs=1, out_dir=str(tmp_path), seed=8, synthetic=True)
    assert metrics["source"] == "synthetic"
    assert metrics["log_loss"] > 0
    assert 0 <= metrics["brier"] <= 1
    assert metrics["onnx_parity"] < 1e-5
    assert (tmp_path / "value-synthetic.onnx").exists()


@pytest.mark.skipif(not demo.SAMPLE_DIR.exists(), reason="sample dataset not present")
@pytest.mark.parametrize("seed", [0, 1])
def test_demo_on_the_committed_sample_beats_the_baseline(tmp_path, seed):
    """The shipped default config must learn something, not just run.

    Guards the demo against a config regression: the full-size net overfits this
    sample within two epochs and scores worse than predicting the base rate.
    """
    metrics = demo.run(out_dir=str(tmp_path), seed=seed)
    assert metrics["source"] == "sample"
    assert metrics["n"] > 1_000
    assert metrics["onnx_parity"] < 1e-5
    assert (tmp_path / "value-sample.onnx").exists()
    assert metrics["log_loss"] < metrics["baseline_log_loss"]
