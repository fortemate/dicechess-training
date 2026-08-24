from dicechess_training import demo


def test_demo_end_to_end(tmp_path):
    metrics = demo.run(n_games=30, epochs=1, out_dir=str(tmp_path), seed=8)
    assert metrics["log_loss"] > 0
    assert 0 <= metrics["brier"] <= 1
    assert metrics["onnx_parity"] < 1e-5
    assert (tmp_path / "value-toy.onnx").exists()
