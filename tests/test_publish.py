"""Write-once publication.

Shared by the serving evidence and the seal because both need the same guarantee, and the first
time it was written twice the second copy kept a defect the first had already lost.
"""

from __future__ import annotations

import contextlib

import pytest

from dicechess_training.publish import stage, write_once


def test_a_taken_destination_is_refused_atomically(tmp_path):
    """`os.link` is the guarantee: a check followed by a rename is a race, a link is not."""
    with contextlib.ExitStack() as stack:
        first = stage(tmp_path / "published.json", "first", stack)
        second = stage(tmp_path / "taken.json", "second", stack)
        taken = tmp_path / "taken.json"
        taken.write_text("published by someone else", encoding="utf-8")
        published = tmp_path / "published.json"

        with pytest.raises(FileExistsError):
            write_once([(first, published), (second, taken)])

        # The document this call did publish is withdrawn; the other call's is untouched.
        assert not published.exists()
        assert taken.read_text(encoding="utf-8") == "published by someone else"


def test_destinations_sharing_a_basename_get_distinct_staged_files(tmp_path):
    """The staged name is fixed, so two destinations cannot collapse onto one staged file."""
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    with contextlib.ExitStack() as stack:
        one = stage(left / "document.json", "one", stack)
        two = stage(right / "document.json", "two", stack)
        assert one != two
        write_once([(one, left / "document.json"), (two, right / "document.json")])
    assert (left / "document.json").read_text(encoding="utf-8") == "one"
    assert (right / "document.json").read_text(encoding="utf-8") == "two"


def test_staging_is_removed_whether_or_not_anything_is_published(tmp_path):
    with contextlib.ExitStack() as stack:
        staged = stage(tmp_path / "out.json", "content", stack)
        directory = staged.parent
        assert directory.is_dir()
    assert not directory.exists()
    assert not (tmp_path / "out.json").exists()
