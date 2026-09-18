"""Write-once publication for the documents this repository's audit tools produce.

Shared rather than repeated, because the two tools that publish audit documents — the serving
evidence and the preregistered seal — need the same guarantee for the same reason, and the first
time it was written twice the second copy kept a defect the first had already lost.

The guarantee is that a destination is never overwritten and a failure leaves nothing behind.
`os.link` is what provides it: it fails when the destination exists, and it fails atomically,
unlike an existence check followed by a rename, which two invocations can both pass before either
writes. It also makes withdrawal exact — a destination this call linked is one no other call could
have linked, so removing it on failure cannot take away a document somebody else published.

A hard link cannot cross a filesystem boundary, so `stage` puts each document beside its own
destination rather than gathering them in one place.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

STAGED_NAME = "document.json"


def stage(destination: Path, text: str, stack: contextlib.ExitStack) -> Path:
    """Write `text` beside `destination`, removed when `stack` unwinds.

    The staged name is fixed rather than taken from the destination, so two destinations that
    happen to share a basename cannot resolve to the same staged file.
    """
    directory = Path(tempfile.mkdtemp(dir=Path(destination).parent, prefix=".publish-staging-"))
    stack.callback(shutil.rmtree, directory, ignore_errors=True)
    staged = directory / STAGED_NAME
    staged.write_text(text, encoding="utf-8")
    return staged


def write_once(pairs: list[tuple[Path, Path]]) -> None:
    """Publish every staged document, or publish none of them.

    Raises `FileExistsError` if any destination is already taken; callers translate that into
    their own refusal so a message never carries a path.
    """
    published: list[Path] = []
    try:
        for staged, destination in pairs:
            os.link(staged, destination)
            published.append(destination)
    except OSError:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
