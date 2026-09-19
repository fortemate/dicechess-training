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
    client = FakeHub(corrupt="rows.json")

    with pytest.raises(hub.HubError, match="differs from the bundle: rows.json"):
        hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)


def test_a_published_copy_that_is_incomplete_is_refused(bundle):
    client = FakeHub(omit="license.txt")

    with pytest.raises(hub.HubError, match="missing license.txt"):
        hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)


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


# --- Model packages (#54) ----------------------------------------------------------------------

PACKAGE_REPO = "owner/models"
PACKAGE_DESTINATION = "run-1/candidate"


@pytest.fixture(scope="module")
def built_package(tmp_path_factory):
    """A real package, built by the packager, so admission is tested against the true artifact."""
    from dicechess_training.candidate import CandidateConfig, build_candidate

    directory = tmp_path_factory.mktemp("hub-package") / "package"
    build_candidate(FIXTURE, directory, CandidateConfig(seed=7, model_id="hub-test-candidate"))
    return directory


@pytest.fixture
def package(tmp_path, built_package):
    """A copy with a stray file beside it, which must not reach the published copy."""
    directory = tmp_path / "package"
    shutil.copytree(built_package, directory)
    (directory / "report.json").write_text("{}", encoding="utf-8")
    return directory


class FakeModelHub(FakeHub):
    """The same fake, for a model repository rather than a dataset one."""

    def __call__(self, command):
        command = list(command)
        if command[0] == "upload":
            self.commands.append(command)
            staged = Path(command[2])
            self.uploaded = {item.name: item.read_bytes() for item in sorted(staged.iterdir())}
            return f"url=https://huggingface.co/{PACKAGE_REPO}/commit/{REVISION}\n"
        if command[0] == "download":
            self.commands.append(command)
            local = Path(command[command.index("--local-dir") + 1])
            published = local / PACKAGE_DESTINATION
            published.mkdir(parents=True)
            for name, payload in self.uploaded.items():
                if name == self.omit:
                    continue
                extra = b" " if name == self.corrupt else b""
                (published / name).write_bytes(payload + extra)
            return ""
        return super().__call__(command)


def test_the_package_contract_matches_the_packagers(built_package):
    """The two definitions are separate on purpose; this is what stops them drifting."""
    from dicechess_training.candidate.build import PACKAGE_FILES

    assert hub.PACKAGE_FILES == PACKAGE_FILES
    assert sorted(p.name for p in built_package.iterdir()) == sorted(hub.PACKAGE_FILES)


def test_a_package_is_admitted_uploaded_and_verified_back(package):
    client = FakeModelHub()

    result = hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)

    assert result["repo_id"] == PACKAGE_REPO
    assert result["path_in_repo"] == PACKAGE_DESTINATION
    assert result["revision"] == REVISION
    assert sorted(result["digests"]) == sorted(hub.PACKAGE_FILES)
    # Exactly the contract: the stray file beside the package never left.
    assert sorted(client.uploaded) == sorted(hub.PACKAGE_FILES)


def test_a_package_goes_to_a_model_repository(package):
    client = FakeModelHub()

    hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)

    for command in client.commands:
        assert "--repo-type" in command
        assert command[command.index("--repo-type") + 1] == "model"
    create = client.commands[0]
    assert create[:2] == ["repos", "create"]
    assert "--private" in create


def test_a_graph_that_does_not_match_its_manifest_is_never_uploaded(package):
    client = FakeModelHub()
    model = package / "model.onnx"
    model.write_bytes(model.read_bytes() + b"\x00")

    with pytest.raises(hub.HubError, match="does not match the manifest's modelSha256"):
        hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)

    assert client.commands == []


def test_a_missing_graph_is_refused(package):
    client = FakeModelHub()
    (package / "model.onnx").unlink()

    with pytest.raises(hub.HubError, match="model.onnx is missing"):
        hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)

    assert client.commands == []


def test_a_schema_with_no_contract_here_is_refused_by_name(package):
    client = FakeModelHub()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["featureSchema"] = "kcp-99"
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(hub.HubError, match="no contract here for feature schema 'kcp-99'"):
        hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)

    assert client.commands == []


def test_a_graph_that_could_not_be_served_is_refused(package):
    """A digest says the files belong together; only the contract says the graph is servable."""
    client = FakeModelHub()
    model = package / "model.onnx"
    model.write_bytes(b"not an onnx graph at all")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["modelSha256"] = hub.kcp13.sha256_of(model)
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(hub.HubError, match="package refused"):
        hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)

    assert client.commands == []


def test_a_published_package_that_came_back_altered_fails_the_run(package):
    client = FakeModelHub(corrupt="model.onnx")

    with pytest.raises(hub.HubError, match="published copy differs from the package: model.onnx"):
        hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)


def test_a_published_package_missing_a_file_fails_the_run(package):
    client = FakeModelHub(omit="model-card.md")

    with pytest.raises(hub.HubError, match="published copy is missing model-card.md"):
        hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)


@pytest.mark.parametrize("destination", ["", "/absolute", "run/../escape"])
def test_a_package_destination_that_could_escape_is_refused(package, destination):
    client = FakeModelHub()

    with pytest.raises(hub.HubError, match="invalid destination path"):
        hub.publish_package(PACKAGE_REPO, package, destination, runner=client)

    assert client.commands == []


def test_the_cli_requires_exactly_one_artifact(package, capsys):
    both = ["--repo", PACKAGE_REPO, "--bundle", str(package), "--package", str(package)]
    assert main([*both, "--path-in-repo", PACKAGE_DESTINATION]) == 1
    assert main(["--repo", PACKAGE_REPO, "--path-in-repo", PACKAGE_DESTINATION]) == 1
    assert capsys.readouterr().out == ""


def test_the_cli_refuses_a_package_without_publishing(package, capsys):
    model = package / "model.onnx"
    model.write_bytes(model.read_bytes() + b"\x00")

    code = main(
        ["--repo", PACKAGE_REPO, "--package", str(package), "--path-in-repo", PACKAGE_DESTINATION]
    )

    assert code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err)["published"] is False
    assert str(package) not in captured.err
    assert captured.out == ""


def test_visibility_is_set_on_the_destination_not_only_requested_at_creation(package):
    """`create --private` decides how a repository is born, not what an existing one is."""
    client = FakeModelHub()

    hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)

    settings = next(c for c in client.commands if c[:2] == ["repos", "settings"])
    assert settings[2] == PACKAGE_REPO
    assert "--private" in settings
    assert settings[settings.index("--repo-type") + 1] == "model"
    # And it happens before anything is uploaded, or it would not have protected the upload.
    names = [c[0] if c[0] != "repos" else " ".join(c[:2]) for c in client.commands]
    assert names.index("repos settings") < names.index("upload")


def test_a_bundle_destination_is_made_private_too(bundle):
    client = FakeHub()

    hub.publish_bundle(REPO, bundle, DESTINATION, runner=client)

    settings = next(c for c in client.commands if c[:2] == ["repos", "settings"])
    assert settings[settings.index("--repo-type") + 1] == "dataset"
    assert "--private" in settings


@pytest.mark.parametrize("missing", ["model-card.md", "manifest.json"])
def test_a_package_missing_any_file_of_the_contract_is_refused(package, missing):
    """The card is not read by admission, but its absence used to escape as a traceback."""
    client = FakeModelHub()
    (package / missing).unlink()

    with pytest.raises(hub.HubError, match=f"{missing} is missing"):
        hub.publish_package(PACKAGE_REPO, package, PACKAGE_DESTINATION, runner=client)

    assert client.commands == []


def test_the_cli_turns_a_missing_card_into_a_refusal_not_a_traceback(package, capsys):
    (package / "model-card.md").unlink()

    code = main(
        ["--repo", PACKAGE_REPO, "--package", str(package), "--path-in-repo", PACKAGE_DESTINATION]
    )

    assert code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err)["published"] is False
    assert "model-card.md" in json.loads(captured.err)["error"]
    assert str(package) not in captured.err
