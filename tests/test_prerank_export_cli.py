"""Export CLI provenance, diagnostics, and side-effect order before refactoring."""

from pathlib import Path

import pytest

from dicechess_training.prerank import __main__ as cli


@pytest.fixture
def exporting(monkeypatch, tmp_path):
    calls = []
    model = object()
    weights = tmp_path / "weights.pt"
    destination = tmp_path / "package"

    def load(path):
        calls.append(("load", path))
        assert path == weights
        return model

    def write(actual, folder, **kwargs):
        assert actual is model
        calls.append(("write", folder, kwargs))
        return {
            "modelId": "export-cli-test",
            "modelRole": "move-prerank",
            "manifestVersion": "1.1.0",
            "featureSchema": "rich-9-v1",
            "engineCompatibility": ">=0.12.0",
            "modelSha256": "digest",
        }

    def probe(actual, graph, engine_version):
        assert actual is model
        assert graph == destination / cli.exporter.MODEL_FILE
        calls.append(("probe", graph, engine_version))
        return 0.0001234

    monkeypatch.setattr(cli.exporter, "load_ranker", load)
    monkeypatch.setattr(cli.exporter, "write_package", write)
    monkeypatch.setattr(cli.exporter, "probe_parity", probe)

    def run(entries):
        return cli.main(
            [
                "export",
                str(weights),
                str(destination),
                "--model-id",
                "export-cli-test",
                "--engine-version",
                "0.12.0",
                "--engine-compatibility",
                ">=0.12.0",
                *[f"--provenance={entry}" for entry in entries],
            ]
        )

    return run, calls


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ([], []),
        (["protocol=sample"], [("protocol", "sample")]),
        (["b=one", "a=two", "b=three"], [("b", "three"), ("a", "two")]),
        (["key=left=right"], [("key", "left=right")]),
        (["=value"], [("", "value")]),
        (["key="], [("key", "")]),
        (["="], [("", "")]),
        ([" key = value "], [(" key ", " value ")]),
        (["автор=Čemisovs", "棋=♟"], [("автор", "Čemisovs"), ("棋", "♟")]),
    ],
)
def test_export_preserves_provenance_and_call_order(exporting, capsys, entries, expected):
    run, calls = exporting
    assert run(entries) == 0
    assert [call[0] for call in calls] == ["load", "write", "probe"]
    kwargs = calls[1][2]
    assert list(kwargs["provenance"].items()) == expected
    assert kwargs["model_id"] == "export-cli-test"
    assert kwargs["engine_version"] == "0.12.0"
    assert kwargs["engine_compatibility"] == ">=0.12.0"
    assert isinstance(calls[1][1], Path)
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == (
        "exported export-cli-test as move-prerank\n"
        "  manifest    1.1.0  schema rich-9-v1\n"
        "  engine      >=0.12.0\n"
        "  modelSha256 digest\n"
        "  probe parity against the checkpoint: 0.000123\n"
    )


@pytest.mark.parametrize(
    ("entries", "index"),
    [
        (["missing"], 0),
        ([""], 0),
        (["ok=1", "missing"], 1),
        (["ok=1", "missing", "also-missing"], 1),
        (["a=1", "b=2", ""], 2),
    ],
)
def test_malformed_provenance_keeps_indexed_error_before_writing(exporting, capsys, entries, index):
    run, calls = exporting
    with pytest.raises(ValueError) as caught:
        run(entries)
    assert str(caught.value) == (
        f"dictionary update sequence element #{index} has length 1; 2 is required"
    )
    assert [call[0] for call in calls] == ["load"]
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_weight_loading_error_precedes_provenance_validation(exporting, monkeypatch, capsys):
    run, calls = exporting

    def fail(path):
        calls.append(("load-failed", path))
        raise ValueError("the checkpoint holds no layers")

    monkeypatch.setattr(cli.exporter, "load_ranker", fail)
    with pytest.raises(ValueError, match="^the checkpoint holds no layers$"):
        run(["malformed"])
    assert [call[0] for call in calls] == ["load-failed"]
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
