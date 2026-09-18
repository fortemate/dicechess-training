"""JSON-only entry point for bundle publication. Failures never print private paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import HubError, publish_bundle


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid CLI arguments")


def _bundle_location(raw: str) -> Path:
    resolved = Path(raw).resolve()
    if not resolved.is_dir():
        raise ValueError("bundle path is not a directory")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = SafeParser(prog="dicechess_training.hub", description="Publish a bundle to the Hub")
    parser.add_argument("--repo", required=True, help="destination dataset repository, owner/name")
    parser.add_argument("--bundle", required=True, type=_bundle_location)
    parser.add_argument(
        "--path-in-repo",
        required=True,
        help="destination directory inside the repository, e.g. <run>/<bundle>",
    )
    try:
        args = parser.parse_args(argv)
        result = publish_bundle(args.repo, args.bundle, args.path_in_repo)
    except (HubError, ValueError) as error:
        # The message is written by this package and carries no local path; the Hub client's own
        # diagnostics have already reached the terminal on stderr.
        print(json.dumps({"published": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"published": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
