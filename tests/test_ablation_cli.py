"""CLI input access and provisional reporting must not depend on output permissions."""

import pytest

from dicechess_training.ablation import __main__ as cli
from dicechess_training.ablation.paths import resolve_read_path
from dicechess_training.ablation.report import _render_header
from dicechess_training.ablation.runner import _safe_protocol_path
from dicechess_training.contracts import SCHEMA_CONTRACTS


def test_cli_accepts_inputs_outside_write_locations(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    protocol = inputs / "protocol.json"
    cost = inputs / "cost.json"
    protocol.write_text("{}")
    cost.write_text("{}")
    monkeypatch.setattr(cli, "ROOT", output)
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(output))
    monkeypatch.chdir(output)
    with pytest.raises(ValueError, match="outside allowed boundaries"):
        cli._safe_write_path(inputs)
    assert _safe_protocol_path(protocol) == protocol

    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {
            "status": "provisional-development",
            "decision": {
                "selected_schema": "S0",
                "selected_schema_id": "kcp-13",
                "gate_results": {},
            },
        }

    monkeypatch.setattr(cli, "run_ablation", run)
    monkeypatch.setattr(cli, "render_markdown_report", lambda report: report["status"])
    cli.main(
        [
            "--protocol",
            str(protocol),
            "--data-dir",
            str(inputs),
            "--extraction-cost",
            str(cost),
            "--output-dir",
            str(output),
        ]
    )
    assert calls == [
        {"protocol_path": protocol, "enriched_base_dir": inputs, "extraction_cost_path": cost}
    ]
    assert (output / "report.md").read_text() == "provisional-development"
    assert not (inputs / "report.md").exists()


def test_input_path_requires_correct_kind(tmp_path):
    path = tmp_path / "file.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="directory"):
        resolve_read_path(path, directory=True)
    with pytest.raises(ValueError, match="file"):
        resolve_read_path(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_read_path(tmp_path / "missing", directory=True)


@pytest.mark.parametrize("key,sid", list(zip(("S0", "S1", "S2"), SCHEMA_CONTRACTS, strict=True)))
def test_every_selected_schema_remains_provisional(key, sid):
    report = {
        "protocol_version": "synthetic-protocol",
        "protocol_sha256": "0" * 64,
        "engine_version": "0.9.3",
    }
    split = dict.fromkeys(
        (
            "train_games",
            "val_games",
            "total_positions",
            "decisive_positions",
            "train_positions",
            "val_positions",
        ),
        1,
    )
    decision = {"selected_schema": key, "selected_schema_id": sid}
    text = "\n".join(_render_header(report, split, decision))
    assert "final owner qualification remains pending" in text
    assert "qualified as the new feature schema" not in text
    assert "sample/playsite-bots-v0" not in text
    assert "synthetic-protocol" in text


def test_cli_fails_closed_when_nothing_is_admissible(tmp_path, monkeypatch, capsys):
    """An inadmissible run still writes its evidence, then exits non-zero (#27)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    protocol = workspace / "protocol.json"
    protocol.write_text("{}")
    cost = workspace / "cost.json"
    cost.write_text("{}")
    monkeypatch.setattr(cli, "ROOT", workspace)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        cli,
        "run_ablation",
        lambda **_: {
            "status": "inadmissible-no-selection",
            "decision": {
                "selected_schema": None,
                "selected_schema_id": None,
                "gate_results": {"S1": False, "S2": False},
                "inadmissible_reason": "every schema scored worse than the reference",
            },
        },
    )
    monkeypatch.setattr(cli, "render_markdown_report", lambda report: report["status"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "--protocol",
                str(protocol),
                "--data-dir",
                str(workspace),
                "--extraction-cost",
                str(cost),
                "--output-dir",
                str(workspace / "out/ablation"),
            ]
        )

    assert exit_info.value.code == 1
    assert "every schema scored worse than the reference" in capsys.readouterr().out
    assert (workspace / "out/ablation/report.md").read_text() == "inadmissible-no-selection"
