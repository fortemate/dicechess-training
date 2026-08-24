from dicechess_training import data, features, onnx_io, train


def test_onnx_export_parity(tmp_path):
    df = data.generate_toy_games(20, seed=7)
    x, y = features.encode_frame(df), df["result"].to_numpy()
    model = train.train_value_model(x, y, epochs=1, seed=7, hidden=32)
    path = str(tmp_path / "value.onnx")
    onnx_io.export_value_model(model, path)
    assert onnx_io.onnx_parity(model, path, n=128, seed=7) < 1e-5
