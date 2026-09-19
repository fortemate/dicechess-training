"""JSON-only entry point for publication. Failures never print private paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import HubError, publish_bundle, publish_package


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid CLI arguments")


def _artifact_location(raw: str) -> Path:
    resolved = Path(raw).resolve()
    if not resolved.is_dir():
        raise ValueError("artifact path is not a directory")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = SafeParser(
        prog="dicechess_training.hub", description="Publish a bundle or a model package to the Hub"
    )
    parser.add_argument("--repo", required=True, help="destination repository, owner/name")
    # Exactly one, because the two differ in what is admitted and in the repository type they
    # publish to; inferring the kind from the directory's contents would guess at the one thing
    # the operator should be explicit about.
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--bundle", type=_artifact_location, help="dataset bundle directory")
    what.add_argument("--package", type=_artifact_location, help="model package directory")
    parser.add_argument(
        "--path-in-repo",
        required=True,
        help="destination directory inside the repository, e.g. <run>/<artifact>",
    )
    try:
        args = parser.parse_args(argv)
        if args.bundle is not None:
            result = publish_bundle(args.repo, args.bundle, args.path_in_repo)
        else:
            result = publish_package(args.repo, args.package, args.path_in_repo)
    except (HubError, ValueError) as error:
        # The message is written by this package and carries no local path; the Hub client's own
        # diagnostics have already reached the terminal on stderr.
        print(json.dumps({"published": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"published": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
