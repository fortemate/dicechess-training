"""Publish an exported bundle to a dataset repository on the Hugging Face Hub.

A bundle is expensive to rebuild and is the unit the qualification protocol binds, so a copy that
is merely *assumed* equal to the exported one is worth very little. The whole value of the copy is
that a later reader can prove it is the same artifact, which is why this module verifies twice:
once before anything is uploaded, by admitting the bundle through the benchmark's own loader, and
once after, against the copy read back from the Hub. Either disagreement refuses the run.

The two checks prove different things and neither replaces the other. The first says the bundle is
internally consistent and admissible — its rows match `rows_sha256`, its terms match
`license_evidence_sha256`, its engine has a committed golden. The second says the transfer did not
alter it. Only the second can catch a truncated upload, and only the first can catch a bundle that
was already wrong before it left.

No separate list of digests is kept here. `manifest.json` already carries them and a seal already
binds the manifest; a second list would be a second thing that can disagree with the first.

The Hub is reached through its command line client rather than a Python dependency. Publishing is
an operator action, and the environment every contributor installs to run the demo should not carry
a client for it. `runner` is the seam that makes this testable with no network, no credential and
no client installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from dicechess_training.contracts import SCHEMA_CONTRACTS, kcp13

from ..benchmark import core

# The bundle contract, exactly. Staging these by name is what keeps a stray file that happens to
# sit beside a bundle — a report, an editor backup — out of the published copy.
BUNDLE_FILES = ("manifest.json", "rows.json", "license.txt")

#: The package contract, exactly, and for the same reason. Restated here rather than imported from
#: the packager, which reaches PyTorch through its training path and would make a publish tool pay
#: for a dependency it never uses; `test_hub.py` asserts the two agree so they cannot drift.
PACKAGE_FILES = ("model.onnx", "manifest.json", "model-card.md")

_REVISION = re.compile(r"/commit/([0-9a-f]{40})")

Runner = Callable[[Sequence[str]], str]


class HubError(RuntimeError):
    """A publication was refused, or the Hub client failed."""


def _run(command: Sequence[str]) -> str:
    """Run the Hub client, returning its stdout.

    stderr is deliberately not captured: the operator sees the client's own diagnostics on their
    terminal, while nothing carrying a local path enters an exception or this tool's JSON output.
    """
    try:
        completed = subprocess.run(["hf", *command], text=True, stdout=subprocess.PIPE, check=False)
    except FileNotFoundError as error:
        raise HubError("hub client not found; see the README for how to install it") from error
    if completed.returncode != 0:
        raise HubError(f"hub client failed: {command[0]} exited {completed.returncode}")
    return completed.stdout


def _stage(source: Path, files: Sequence[str]) -> tuple[Path, Callable[[], None]]:
    """Materialize exactly the named contract in a directory of its own.

    Hard links are tried first so that staging a large artifact costs no bytes; the staging
    directory is created beside the source to keep the link on one filesystem. A copy is the
    fallback, since a link is an optimization and not part of the guarantee.
    """
    directory = Path(tempfile.mkdtemp(dir=source.parent, prefix=".hub-staging-"))

    def cleanup() -> None:
        shutil.rmtree(directory, ignore_errors=True)

    try:
        for name in files:
            try:
                os.link(source / name, directory / name)
            except OSError:
                shutil.copy2(source / name, directory / name)
    except Exception:
        cleanup()
        raise
    return directory, cleanup


def digests(directory: Path, files: Sequence[str] = BUNDLE_FILES) -> dict[str, str]:
    return {name: kcp13.sha256_of(directory / name) for name in files}


def admit(bundle: Path) -> dict:
    """Refuse a bundle the benchmark itself would not load.

    Publishing something the protocol cannot admit would create an artifact that looks citable and
    is not, which is worse than having no copy at all.
    """
    try:
        manifest, _ = core.load_dataset(bundle)
    except (ValueError, KeyError, OSError) as error:
        raise HubError(f"bundle refused: {error}") from error
    return manifest


def admit_package(package: Path) -> dict:
    """Refuse a package that could not be served, before any of it is uploaded.

    A bundle is admitted by the benchmark's own loader. A package has no equivalent loader, but it
    is self-describing: the manifest states what its own graph should hash to, so verification
    leans on that claim rather than on a digest list kept beside it — a second list would be a
    second thing that can disagree with the first.

    Two checks, and they answer different questions. The digest says the three files belong
    together. The contract says the graph could actually be served under the schema it declares,
    which a digest cannot tell you: a package whose manifest and graph agree perfectly can still
    expose the wrong tensors.
    """
    try:
        manifest = core.read_json(package / "manifest.json")
    except (ValueError, KeyError, OSError) as error:
        raise HubError(f"package refused: unreadable manifest: {error}") from error

    schema = manifest.get("featureSchema")
    contract = SCHEMA_CONTRACTS.get(schema)
    if contract is None:
        # Named rather than skipped: a package for a schema this repository cannot describe is the
        # case most worth stopping, because nothing downstream would notice it was never checked.
        raise HubError(f"package refused: no contract here for feature schema {schema!r}")

    model = package / "model.onnx"
    declared = str(manifest.get("modelSha256", ""))
    if not model.is_file():
        raise HubError("package refused: model.onnx is missing")
    if kcp13.sha256_of(model) != declared.lower():
        raise HubError("package refused: model.onnx does not match the manifest's modelSha256")

    # Only a serving contract knows how to read tensor names out of a manifest; an ablation schema
    # has none, and its graph is admitted under the contract's own defaults.
    names = getattr(contract, "manifest_tensor_names", None)
    try:
        contract.validate_onnx_contract(model, *(names(manifest) if names else ()))
    except Exception as error:
        raise HubError(f"package refused: {error}") from error
    return manifest


def _publish(
    repo_id: str,
    source: Path,
    path_in_repo: str,
    *,
    files: Sequence[str],
    repo_type: str,
    label: str,
    runner: Runner,
) -> dict:
    """Create, stage, upload and read back — the part that is the same whatever is published.

    Shared rather than repeated, because the guarantee is the same one: what landed is what left,
    and a second copy of it would be a second thing to keep correct.
    """
    expected = digests(source, files)

    # Private is not a default to be overridden by a flag here: a bundle ships an owner-controlled
    # data policy and a package ships trained weights, so a public destination would be a
    # disclosure rather than a configuration choice. An existing repository keeps whatever
    # visibility it already has, which is why the caller is told the destination back.
    runner(["repos", "create", repo_id, "--repo-type", repo_type, "--private", "--exist-ok"])

    staged, cleanup = _stage(source, files)
    try:
        output = runner(
            [
                "upload",
                repo_id,
                str(staged),
                path_in_repo,
                "--repo-type",
                repo_type,
                "--commit-message",
                f"Publish {label} {path_in_repo}",
            ]
        )
    finally:
        cleanup()

    with tempfile.TemporaryDirectory(prefix="hub-verify-") as workspace:
        runner(
            [
                "download",
                repo_id,
                "--repo-type",
                repo_type,
                "--local-dir",
                workspace,
                "--include",
                f"{path_in_repo}/*",
            ]
        )
        published = Path(workspace) / path_in_repo
        for name in files:
            if not (published / name).is_file():
                raise HubError(f"published copy is missing {name}")
        actual = digests(published, files)

    for name in files:
        if actual[name] != expected[name]:
            raise HubError(f"published copy differs from the {label}: {name}")

    matches = _REVISION.findall(output or "")
    return {
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "revision": matches[-1] if matches else None,
        "digests": expected,
    }


def _destination(source: Path, path_in_repo: str, kind: str) -> Path:
    source = Path(source)
    if not source.is_dir():
        raise HubError(f"{kind} is not a directory")
    if not path_in_repo or path_in_repo.startswith("/") or ".." in Path(path_in_repo).parts:
        raise HubError("invalid destination path")
    return source


def publish_bundle(
    repo_id: str,
    bundle: str | Path,
    path_in_repo: str,
    *,
    runner: Runner = _run,
) -> dict:
    """Publish `bundle` under `path_in_repo` of the private dataset repository `repo_id`.

    Returns the destination, the revision the upload produced and the digests that were verified on
    both sides, so a caller can record what it published without re-deriving it.
    """
    bundle = _destination(bundle, path_in_repo, "bundle")
    admit(bundle)
    return _publish(
        repo_id,
        bundle,
        path_in_repo,
        files=BUNDLE_FILES,
        repo_type="dataset",
        label="bundle",
        runner=runner,
    )


def publish_package(
    repo_id: str,
    package: str | Path,
    path_in_repo: str,
    *,
    runner: Runner = _run,
) -> dict:
    """Publish `package` under `path_in_repo` of the private **model** repository `repo_id`.

    The same guarantee the bundle path gives, with the admission a package needs instead of the
    one a bundle needs, and a model repository rather than a dataset one.
    """
    package = _destination(package, path_in_repo, "package")
    admit_package(package)
    return _publish(
        repo_id,
        package,
        path_in_repo,
        files=PACKAGE_FILES,
        repo_type="model",
        label="package",
        runner=runner,
    )
