"""Issuing the preregistered seal.

The seal is the last lock before a single-use holdout is opened, so the tests that matter are the
ones proving it refuses at issue time everything qualification would refuse later — and that what
it does issue is accepted by the command it exists to unlock.
"""

from __future__ import annotations

import json
import shutil

import pytest

from dicechess_training.benchmark import core
from dicechess_training.benchmark.splits import position_key
from dicechess_training.candidate import CandidateConfig, build_candidate
from dicechess_training.contracts import kcp13
from dicechess_training.seal import SealError, build_seal, publish_seal
from dicechess_training.seal.__main__ import main

FIXTURE = core.ROOT / "tests/fixtures/benchmark"
WORKLOAD = "c" * 64


def _write_bundle(directory, rows, kind):
    shutil.copytree(FIXTURE, directory)
    (directory / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    manifest = core.read_json(directory / "manifest.json")
    manifest["kind"] = kind
    manifest["rows_sha256"] = kcp13.sha256_of(directory / "rows.json")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    """A development and a final bundle that share no position.

    The fixture's 240 rows cover only 18 distinct positions, each repeated, so a disjoint pair has
    to be cut by position rather than by row — which is the same thing the real sealed export does.
    """
    root = tmp_path_factory.mktemp("seal")
    rows = core.read_json(FIXTURE / "rows.json")
    keys = sorted({position_key(row) for row in rows})
    development_keys = set(keys[: len(keys) // 2])
    development = [row for row in rows if position_key(row) in development_keys]
    final = [
        {
            **row,
            "id": "sealed-" + row["id"],
            "group_id": "sealed-" + row["group_id"],
            "game_id": "sealed-" + row["game_id"],
        }
        for row in rows
        if position_key(row) not in development_keys
    ]
    return (
        _write_bundle(root / "dev", development, "public-licensed"),
        _write_bundle(root / "final", final, "owner-controlled"),
        root,
    )


@pytest.fixture(scope="module")
def candidate(bundles):
    development, _, root = bundles
    directory = root / "candidate"
    build_candidate(development, directory, CandidateConfig(seed=3, model_id="seal-test-candidate"))
    return directory


def _seal(candidate_dir, development, final, **overrides):
    arguments = {
        "concurrency_workload_sha256": WORKLOAD,
        "latency_p95_ms": 25.0,
        "rss_mb": 512.0,
    }
    arguments.update(overrides)
    return build_seal(candidate_dir, development, final, **arguments)


def test_every_derivable_field_comes_from_the_artifacts(bundles, candidate):
    development, final, _ = bundles
    seal = _seal(candidate, development, final)
    assert seal["schema"] == "playground-seal-v1"
    assert seal["implementation_sha256"] == core.implementation_digest()
    assert seal["benchmark_sha256"] == core.digest(core.load_protocol())
    assert seal["development_dataset_sha256"] == core.digest(
        core.read_json(development / "manifest.json")
    )
    assert seal["final_dataset_sha256"] == core.digest(core.read_json(final / "manifest.json"))
    assert seal["candidate_manifest_sha256"] == core.digest(
        core.read_json(candidate / "manifest.json")
    )
    assert seal["accepted_references"] == []
    assert seal["promotion_reference"] == "no-information"
    assert seal["concurrency_workload_sha256"] == WORKLOAD
    assert seal["serving_limits"] == {"latency_p95_ms": 25.0, "rss_mb": 512.0}


def test_the_command_it_unlocks_accepts_what_this_issues(tmp_path, bundles, candidate):
    """The only acceptance test that counts: `prepare_final` gets past every binding check."""
    development, final, _ = bundles
    path = tmp_path / "seal.json"
    digest = publish_seal(_seal(candidate, development, final), path)
    data, rows = core.load_dataset(final)
    context = core.prepare_final(
        data,
        rows,
        core.load_candidate(
            candidate, core.read_json(development / "manifest.json"), core.load_protocol()
        ),
        core.load_protocol(),
        development,
        path,
        digest,
    )
    assert context.seal["schema"] == "playground-seal-v1"
    assert context.reasons == []


def test_the_published_digest_is_the_digest_of_the_file(tmp_path, bundles, candidate):
    development, final, _ = bundles
    path = tmp_path / "seal.json"
    assert publish_seal(_seal(candidate, development, final), path) == kcp13.sha256_of(path)


def test_a_final_bundle_that_repeats_a_position_never_consumes_a_seal(tmp_path, bundles, candidate):
    """Fresh games over positions development has already seen — the trap the exporter exists for.

    Group, game and root disjointness holds here, so only the exact-position rule can catch it.
    """
    development, _, _ = bundles
    rows = [
        {
            **row,
            "id": "reused-" + row["id"],
            "group_id": "reused-" + row["group_id"],
            "game_id": "reused-" + row["game_id"],
        }
        for row in core.read_json(development / "rows.json")
    ]
    overlapping = _write_bundle(tmp_path / "overlapping", rows, "owner-controlled")
    with pytest.raises(SealError, match="repeats a position development has already seen"):
        _seal(candidate, development, overlapping)


def test_a_final_bundle_sharing_a_group_is_refused(tmp_path, bundles, candidate):
    development, final, _ = bundles
    rows = core.read_json(final / "rows.json")
    rows[0]["group_id"] = core.read_json(development / "rows.json")[0]["group_id"]
    shared = _write_bundle(tmp_path / "shared-group", rows, "owner-controlled")
    with pytest.raises(SealError, match="shares a group, game or root"):
        _seal(candidate, development, shared)


def test_a_final_bundle_that_is_not_owner_controlled_is_refused(tmp_path, bundles, candidate):
    development, final, _ = bundles
    borrowed = _write_bundle(
        tmp_path / "borrowed", core.read_json(final / "rows.json"), "public-licensed"
    )
    with pytest.raises(SealError, match="must be owner-controlled"):
        _seal(candidate, development, borrowed)


def test_synthetic_development_cannot_qualify_a_model(tmp_path, bundles, candidate):
    development, final, _ = bundles
    synthetic = _write_bundle(
        tmp_path / "synthetic", core.read_json(development / "rows.json"), "synthetic"
    )
    with pytest.raises(SealError, match="synthetic development"):
        _seal(candidate, synthetic, final)


def test_a_candidate_trained_on_other_data_is_refused(tmp_path, bundles, candidate):
    """The seal binds a candidate to a development bundle, so a foreign one cannot be sealed.

    One row fewer is enough: `train_identity` digests the manifest and the training group set.
    """
    development, final, _ = bundles
    rows = core.read_json(development / "rows.json")[:-1]
    altered = _write_bundle(tmp_path / "altered-development", rows, "public-licensed")
    with pytest.raises(SealError, match="not trained on this development bundle"):
        _seal(candidate, altered, final)


def test_naming_a_comparator_with_no_accepted_reference_is_refused(bundles, candidate):
    development, final, _ = bundles
    with pytest.raises(SealError, match="no accepted reference for the comparator to name"):
        _seal(candidate, development, final, promotion_reference="a" * 64)


def test_an_accepted_reference_forces_a_deployable_comparator(bundles, candidate, tmp_path):
    """`reference_predictions` refuses `no-information` once a model has been accepted."""
    development, final, _ = bundles
    reference = tmp_path / "reference"
    build_candidate(development, reference, CandidateConfig(seed=9, model_id="accepted-reference"))
    digest = core.digest(core.read_json(reference / "manifest.json"))

    with pytest.raises(SealError, match="cannot be the no-information one"):
        _seal(candidate, development, final, reference_dirs=[reference])
    with pytest.raises(SealError, match="cannot be the no-information one"):
        _seal(
            candidate,
            development,
            final,
            reference_dirs=[reference],
            promotion_reference="no-information",
        )
    with pytest.raises(SealError, match="not an accepted reference"):
        _seal(
            candidate, development, final, reference_dirs=[reference], promotion_reference="f" * 64
        )

    seal = _seal(
        candidate, development, final, reference_dirs=[reference], promotion_reference=digest
    )
    assert seal["accepted_references"] == [digest]
    assert seal["promotion_reference"] == digest


def test_the_same_reference_twice_is_refused(bundles, candidate, tmp_path):
    development, final, _ = bundles
    reference = tmp_path / "twice"
    build_candidate(development, reference, CandidateConfig(seed=9, model_id="accepted-reference"))
    digest = core.digest(core.read_json(reference / "manifest.json"))
    with pytest.raises(SealError, match="supplied twice"):
        _seal(
            candidate,
            development,
            final,
            reference_dirs=[reference, reference],
            promotion_reference=digest,
        )


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf"), True, "fast", None])
def test_an_unusable_serving_limit_is_refused(bundles, candidate, limit):
    development, final, _ = bundles
    with pytest.raises(SealError, match="serving limit"):
        _seal(candidate, development, final, latency_p95_ms=limit)


def test_an_unusable_workload_digest_is_refused(bundles, candidate):
    development, final, _ = bundles
    with pytest.raises(SealError, match="workload digest is unusable"):
        _seal(candidate, development, final, concurrency_workload_sha256="not-a-digest")


def test_an_existing_seal_is_never_overwritten(tmp_path, bundles, candidate):
    development, final, _ = bundles
    path = tmp_path / "seal.json"
    path.write_text("an earlier preregistration", encoding="utf-8")
    seal = _seal(candidate, development, final)
    with pytest.raises(SealError, match="already exists"):
        publish_seal(seal, path)
    assert path.read_text(encoding="utf-8") == "an earlier preregistration"


def test_cli_reports_the_digest_to_retain_and_names_no_artifact(
    tmp_path, bundles, candidate, capsys
):
    development, final, _ = bundles
    path = tmp_path / "cli-seal.json"
    code = main(
        [
            "--candidate",
            str(candidate),
            "--development-data",
            str(development),
            "--final-data",
            str(final),
            "--concurrency-workload-sha256",
            WORKLOAD,
            "--latency-p95-ms",
            "25",
            "--rss-mb",
            "512",
            "--output",
            str(path),
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["seal_sha256"] == kcp13.sha256_of(path)
    assert printed["promotion_reference"] == "no-information"
    assert "retain" in printed
    # Limits and identities stay in the file the owner keeps, never in what the command prints.
    text = json.dumps(printed)
    assert "512" not in text and "seal-test-candidate" not in text


def test_cli_writes_nothing_when_it_refuses(tmp_path, bundles, candidate, capsys):
    development, final, _ = bundles
    path = tmp_path / "refused.json"
    code = main(
        [
            "--candidate",
            str(candidate),
            "--development-data",
            str(development),
            "--final-data",
            str(development),
            "--concurrency-workload-sha256",
            WORKLOAD,
            "--latency-p95-ms",
            "25",
            "--rss-mb",
            "512",
            "--output",
            str(path),
        ]
    )
    assert code == 2
    assert not path.exists()
    assert json.loads(capsys.readouterr().out)["error"] == "seal-issue-failed"
