"""Publication is exercised against a fake Hub client: no network, no credential, no client."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dicechess_training import hub
from dicechess_training.benchmark import core
from dicechess_training.hub.__main__ import main

FIXTURE = core.ROOT / "tests/fixtures/benchmark"
REPO = "owner/bundles"
DESTINATION = "run-1/bundle-dev"
REVISION = "a" * 40


@pytest.fixture
def bundle(tmp_path):
    """A bundle holding exactly the contract, plus a stray file that must not be published."""
    directory = tmp_path / "bundle"
    directory.mkdir()
    for name in hub.BUNDLE_FILES:
        shutil.copy2(FIXTURE / name, directory / name)
    (directory / "report.json").write_text("{}", encoding="utf-8")
    return directory


class FakeHub:
    """Records commands and serves a download from what the upload was handed.

    The upload is captured eagerly, because the real client has the bytes server-side by the time
    a download asks for them while the staging directory is already gone — a fake that read the
    staging directory lazily would be modelling a Hub that does not exist.
    """

    def __init__(self, *, corrupt: str | None = None, omit: str | None = None):
        self.commands: list[list[str]] = []
        self.uploaded: dict[str, bytes] = {}
        self.corrupt = corrupt
        self.omit = omit

    def __call__(self, command):
        command = list(command)
        self.commands.append(command)
        if command[0] == "upload":
            staged = Path(command[2])
            self.uploaded = {item.name: item.read_bytes() for item in sorted(staged.iterdir())}
            return f"url=https://huggingface.co/datasets/{REPO}/commit/{REVISION}\n"
        if command[0] == "download":
            local = Path(command[command.index("--local-dir") + 1])
            published = local / DESTINATION
            published.mkdir(parents=True)
            for name, payload in self.uploaded.items():
                if name == self.omit:
                    continue
                extra = b" " if name == self.corrupt else b""
                (published / name).write_bytes(payload + extra)
        return ""


def test_a_bundle_is_admitted_uploaded_and_verified_back(bundle):
    client = FakeHub()

    result = hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)

    assert result["repo_id"] == REPO
    assert result["path_in_repo"] == DESTINATION
    assert result["revision"] == REVISION
    assert set(result["digests"]) == set(hub.BUNDLE_FILES)
    assert (
        result["digests"]["rows.json"]
        == json.loads((bundle / "manifest.json").read_text())["rows_sha256"]
    )


def test_only_the_bundle_contract_is_published(bundle):
    client = FakeHub()

    hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)

    # The upload is handed a staged directory, not the bundle: a stray file beside a bundle is
    # exactly how something unreviewed reaches a published artifact.
    assert sorted(client.uploaded) == sorted(hub.BUNDLE_FILES)


def test_the_destination_repository_is_created_private(bundle):
    client = FakeHub()

    hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)

    create = next(command for command in client.commands if command[:2] == ["repos", "create"])
    assert "--private" in create
    assert "--exist-ok" in create
    assert create.index("repos") < client.commands.index(
        next(command for command in client.commands if command[0] == "upload")
    )


def test_staging_leaves_nothing_behind(bundle):
    hub.publish_bundle(REPO, bundle, DESTINATION, runner=FakeHub())

    assert [item.name for item in bundle.parent.iterdir()] == ["bundle"]


def test_a_bundle_whose_rows_do_not_match_its_manifest_is_refused(bundle):
    rows = bundle / "rows.json"
    rows.write_bytes(rows.read_bytes() + b" ")
    client = FakeHub()

    with pytest.raises(hub.HubError, match="bundle refused"):
        hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)

    assert client.commands == []


def test_a_published_copy_that_differs_is_refused(bundle):
    with pytest.raises(hub.HubError, match="differs from the bundle: rows.json"):
        hub.publish_bundle(REPO, bundle, DESTINATION, runner=FakeHub(corrupt="rows.json"))


def test_a_published_copy_that_is_incomplete_is_refused(bundle):
    with pytest.raises(hub.HubError, match="missing license.txt"):
        hub.publish_bundle(REPO, bundle, DESTINATION, runner=FakeHub(omit="license.txt"))


def test_publishing_the_same_bundle_twice_verifies_both_times(bundle):
    client = FakeHub()

    first = hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)
    second = hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)

    assert first["digests"] == second["digests"]


@pytest.mark.parametrize("destination", ["", "/absolute", "run/../escape"])
def test_a_destination_that_could_escape_its_directory_is_refused(bundle, destination):
    client = FakeHub()

    with pytest.raises(hub.HubError, match="invalid destination path"):
        hub.publish_bundle(REPO, bundle, destination, runner=client)

    assert client.commands == []


def test_the_cli_reports_a_refusal_without_publishing(bundle, capsys):
    rows = bundle / "rows.json"
    rows.write_bytes(rows.read_bytes() + b" ")

    code = main(["--repo", REPO, "--bundle", str(bundle), "--path-in-repo", DESTINATION])

    assert code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err)["published"] is False
    assert str(bundle) not in captured.err
    assert captured.out == ""


def test_the_cli_refuses_a_bundle_path_that_is_not_a_directory(tmp_path, capsys):
    missing = tmp_path / "nowhere"

    code = main(["--repo", REPO, "--bundle", str(missing), "--path-in-repo", DESTINATION])

    assert code == 1
    assert str(missing) not in capsys.readouterr().err
