"""The source identity binds full relative paths and current file bytes."""

from pathlib import Path

import pytest

from dicechess_training import runtime as execution

GOLDEN = "36227a63b016128dc5d123680d33938b3d38ed29a585f6bc065c4007741daf7a"


@pytest.fixture
def package(monkeypatch, tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    for name, content in {
        ".hidden.py": "hidden\n",
        "a/z.py": "nested\n",
        "a.py": "root\n",
        "plain.py": "plain\n",
        "棋.py": "unicode\n",
        "ignored.pyc": "bytecode",
        "notes.txt": "not source",
    }.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    monkeypatch.setattr(execution, "__file__", str(root / "runtime.py"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    return root, execution.Runtime(execution.RunOptions(device="cpu"))


@pytest.mark.parametrize("location", ["absolute", "relative", "current-directory"])
def test_source_identity_binds_relative_names_in_path_order(package, monkeypatch, location):
    root, context = package
    if location == "relative":
        monkeypatch.chdir(root.parent)
        monkeypatch.setattr(execution, "__file__", "package/runtime.py")
    elif location == "current-directory":
        monkeypatch.chdir(root)
        monkeypatch.setattr(execution, "__file__", "runtime.py")
    environment = context.environment()
    assert environment["source_sha256"] == GOLDEN
    assert environment["world_size"] == 1
    assert environment["device"] == "cpu"
    assert environment["device_name"] == "cpu"


def test_source_identity_does_not_follow_directory_symlinks(package):
    root, context = package
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "extra.py").write_text("not in this package\n", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    assert context.environment()["source_sha256"] == GOLDEN


def test_source_identity_reads_python_file_symlink_at_its_package_name(package):
    root, context = package
    outside = root.parent / "external.py"
    outside.write_text("external\n", encoding="utf-8")
    (root / "linked.py").symlink_to(outside)
    assert context.environment()["source_sha256"] == (
        "93842caed4d636da6b9bdc27a7743ade814fc43f6c7a1979b0437566a80d8e75"
    )


def test_source_identity_observes_edits_and_new_files_without_changing_other_metadata(package):
    root, context = package
    original = context.environment()
    (root / "plain.py").write_text("changed\n", encoding="utf-8")
    edited = context.environment()
    assert edited == original | {
        "source_sha256": "5dcb823a8d9f1251a3e261a4c7ea0518f8169912093b6f3f72fa7d801d26e178"
    }
    (root / "added.py").write_text("new\n", encoding="utf-8")
    assert context.environment() == original | {
        "source_sha256": "7f566e0a6e9ab6f1fa58326a1f382cd884ec3750fdd47497dece357059d8518c"
    }


def test_matching_directory_keeps_its_original_read_error(package):
    root, context = package
    (root / "directory.py").mkdir()
    with pytest.raises(IsADirectoryError):
        context.environment()


def test_source_read_failure_is_not_silently_skipped(package, monkeypatch):
    root, context = package
    failure = OSError("authored source read failure")
    original = Path.read_bytes

    def read(path):
        if path == root / "a.py":
            raise failure
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(OSError) as caught:
        context.environment()
    assert caught.value is failure
