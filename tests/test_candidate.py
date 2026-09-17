"""The candidate packager: admissibility proved through the real consumer, and fail-closed refusals.

Nothing here asserts that a candidate is *good*: the benchmark judges quality, and these tests
only prove that a produced package is the artifact the benchmark agrees to read, and that a
package which would be wrong is never written at all.
"""

from __future__ import annotations

import json
import shutil

import pytest

from dicechess_training.benchmark import core
from dicechess_training.benchmark.splits import split_for
from dicechess_training.candidate import CandidateConfig, build_candidate
from dicechess_training.candidate.__main__ import main
from dicechess_training.candidate.build import (
    MANIFEST_FILE,
    MODEL_CARD_FILE,
    MODEL_FILE,
    CandidateError,
    group_value,
)
from dicechess_training.contracts import kcp13

FIXTURE = core.ROOT / "tests/fixtures/benchmark"


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    directory = tmp_path_factory.mktemp("candidate") / "package"
    summary = build_candidate(
        FIXTURE, directory, CandidateConfig(seed=11, model_id="packager-test-candidate")
    )
    return directory, summary


def test_group_value_agrees_with_the_benchmark_split():
    for i in range(500):
        group = f"group-{i}"
        value = group_value(group)
        expected = "train" if value < 8000 else ("validation" if value < 9000 else "test")
        assert split_for(group) == expected


def test_package_holds_exactly_the_expected_artifacts(package):
    directory, _ = package
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [MODEL_FILE, MANIFEST_FILE, MODEL_CARD_FILE]
    )


def test_the_benchmark_admits_the_package(package):
    directory, _ = package
    data_manifest, rows = core.load_dataset(FIXTURE)
    manifest = core.load_candidate(directory, data_manifest, core.load_protocol())
    core.check_training_identity(manifest, data_manifest, rows)
    assert manifest["featureSchema"] == kcp13.SCHEMA_ID


def test_the_benchmark_evaluates_the_package_end_to_end(package):
    directory, _ = package
    report = core.evaluate(FIXTURE, directory, mode="development")
    assert report["mode"] == "development"
    assert report["candidate_manifest_sha256"] == core.digest(
        core.read_json(directory / MANIFEST_FILE)
    )
    # A development run never promotes; the point is that the package was read and judged.
    assert "development-only" in report["decision"]["reasons"]


def test_provenance_carries_every_digest_the_benchmark_requires(package):
    _, summary = package
    provenance = summary["provenance"]
    for key in (
        "training_data_sha256",
        "training_groups_sha256",
        "config_sha256",
        "benchmark_sha256",
        "code_sha256",
    ):
        assert len(provenance[key]) == 64
    assert provenance["benchmark_sha256"] == core.digest(core.load_protocol())
    assert (
        provenance["engine_version"] == core.read_json(FIXTURE / "manifest.json")["engine_version"]
    )
    assert isinstance(provenance["seed"], int)


def test_parity_is_within_the_contract_tolerance(package):
    _, summary = package
    assert summary["parity"]["golden_max_abs_diff"] <= 1e-6
    assert summary["parity"]["held_out_max_abs_diff"] <= 1e-6


def test_budget_is_selected_on_the_inner_split_only(package):
    _, summary = package
    selection = summary["epoch_selection"]
    assert selection["selected_epochs"] in (5, 10, 20, 40)
    assert len(selection["candidates"]) == 4
    assert summary["inner_tuning_rows"] > 0
    assert summary["inner_tuning_rows"] < summary["training_rows"]


def test_model_card_records_digests_and_claims_no_verdict(package):
    directory, summary = package
    card = (directory / MODEL_CARD_FILE).read_text(encoding="utf-8")
    assert summary["manifest"]["modelSha256"] in card
    assert summary["provenance"]["code_sha256"] in card
    assert "promotion stays an owner decision" in card
    assert str(FIXTURE) not in card


def test_a_second_build_refuses_to_overwrite_a_package(package, tmp_path):
    directory, _ = package
    target = tmp_path / "again"
    shutil.copytree(directory, target)
    with pytest.raises(CandidateError, match="already exists"):
        build_candidate(FIXTURE, target, CandidateConfig(seed=11, model_id="second"))


def test_calibration_in_the_manifest_is_refused(tmp_path):
    with pytest.raises(CandidateError, match="embedded in the graph"):
        build_candidate(
            FIXTURE,
            tmp_path / "calibrated",
            CandidateConfig(seed=11, model_id="c", calibration={"temperature": 2.0}),
        )
    assert not (tmp_path / "calibrated").exists() or not any((tmp_path / "calibrated").iterdir())


def test_a_range_excluding_the_dataset_engine_is_refused(tmp_path):
    with pytest.raises((CandidateError, kcp13.ContractError)):
        build_candidate(
            FIXTURE,
            tmp_path / "narrow",
            CandidateConfig(seed=11, model_id="c", engine_compatibility=">=0.13.0"),
        )
    assert not (tmp_path / "narrow").exists() or not any((tmp_path / "narrow").iterdir())


def test_a_dataset_the_benchmark_refuses_never_trains(tmp_path):
    corrupt = tmp_path / "data"
    shutil.copytree(FIXTURE, corrupt)
    manifest = core.read_json(corrupt / "manifest.json")
    manifest["engine_version"] = "0.0.1"
    (corrupt / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="unverified engine"):
        build_candidate(corrupt, tmp_path / "out", CandidateConfig(seed=11, model_id="c"))
    assert not (tmp_path / "out").exists()


def test_a_tampered_model_loses_benchmark_admission(package, tmp_path):
    directory, _ = package
    tampered = tmp_path / "tampered"
    shutil.copytree(directory, tampered)
    with (tampered / MODEL_FILE).open("ab") as model:
        model.write(b"\0")
    data_manifest, _ = core.load_dataset(FIXTURE)
    with pytest.raises(kcp13.ContractError, match="SHA-256 mismatch"):
        core.load_candidate(tampered, data_manifest, core.load_protocol())


def test_cli_reports_digests_only(tmp_path, capsys):
    code = main(
        [
            "--data",
            str(FIXTURE),
            "--output",
            str(tmp_path / "cli"),
            "--seed",
            "23",
            "--model-id",
            "cli-candidate",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert str(FIXTURE) not in printed and str(tmp_path) not in printed
    report = json.loads(printed)
    assert report["schema"] == "playground-candidate-v1"
    assert len(report["model_sha256"]) == 64


def test_cli_fails_closed_without_leaking_input(tmp_path, capsys):
    code = main(
        [
            "--data",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out"),
            "--seed",
            "1",
            "--model-id",
            "x",
        ]
    )
    assert code == 2
    printed = capsys.readouterr().out
    assert json.loads(printed) == {
        "schema": "playground-error-v1",
        "error": "candidate-build-failed",
    }
