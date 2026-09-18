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

from dicechess_training.contracts import kcp13

from ..benchmark import core

# The bundle contract, exactly. Staging these by name is what keeps a stray file that happens to
# sit beside a bundle — a report, an editor backup — out of the published copy.
BUNDLE_FILES = ("manifest.json", "rows.json", "license.txt")

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


def _stage(bundle: Path) -> tuple[Path, Callable[[], None]]:
    """Materialize exactly the bundle contract in a directory of its own.

    Hard links are tried first so that staging a large bundle costs no bytes; the staging directory
    is created beside the bundle to keep the link on one filesystem. A copy is the fallback, since
    a link is an optimization and not part of the guarantee.
    """
    directory = Path(tempfile.mkdtemp(dir=bundle.parent, prefix=".hub-staging-"))

    def cleanup() -> None:
        shutil.rmtree(directory, ignore_errors=True)

    try:
        for name in BUNDLE_FILES:
            try:
                os.link(bundle / name, directory / name)
            except OSError:
                shutil.copy2(bundle / name, directory / name)
    except Exception:
        cleanup()
        raise
    return directory, cleanup


def digests(directory: Path) -> dict[str, str]:
    return {name: kcp13.sha256_of(directory / name) for name in BUNDLE_FILES}


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
    bundle = Path(bundle)
    if not bundle.is_dir():
        raise HubError("bundle is not a directory")
    if not path_in_repo or path_in_repo.startswith("/") or ".." in Path(path_in_repo).parts:
        raise HubError("invalid destination path")

    admit(bundle)
    expected = digests(bundle)

    # Private is not a default to be overridden by a flag here: a bundle ships an owner-controlled
    # data policy, and a public destination would be a disclosure rather than a configuration
    # choice. An existing repository keeps whatever visibility it already has, which is why the
    # caller is told the destination back and is expected to look.
    runner(["repos", "create", repo_id, "--repo-type", "dataset", "--private", "--exist-ok"])

    staged, cleanup = _stage(bundle)
    try:
        output = runner(
            [
                "upload",
                repo_id,
                str(staged),
                path_in_repo,
                "--repo-type",
                "dataset",
                "--commit-message",
                f"Publish bundle {path_in_repo}",
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
                "dataset",
                "--local-dir",
                workspace,
                "--include",
                f"{path_in_repo}/*",
            ]
        )
        published = Path(workspace) / path_in_repo
        for name in BUNDLE_FILES:
            if not (published / name).is_file():
                raise HubError(f"published copy is missing {name}")
        actual = digests(published)

    for name in BUNDLE_FILES:
        if actual[name] != expected[name]:
            raise HubError(f"published copy differs from the bundle: {name}")

    matches = _REVISION.findall(output or "")
    return {
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "revision": matches[-1] if matches else None,
        "digests": expected,
    }
