"""Read-only CLI inputs may reside outside the permitted report output locations."""

from pathlib import Path


def resolve_read_path(path: Path, *, directory: bool = False) -> Path:
    resolved = path.resolve()
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"Input {kind} does not exist: {resolved}")
    return resolved
