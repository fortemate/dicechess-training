"""The two ends of the pre-ranker's corpus: choose the roots, then admit what came back.

Between them sits the generator in `dicechess-hunter`, which is where the teacher lives:

```text
python -m dicechess_training.prerank roots  <shards-dir> <out-dir> --limit 5000
#   ... mise run prerank:corpus -- <out-dir>/roots.tsv <profile.json> <corpus-dir>   (hunter)
python -m dicechess_training.prerank pack   <corpus-dir> --license-file <terms.txt> --license "..."
```

Failures print what went wrong and never a path, for the same reason the rest of the publication
boundary does not: a corpus is private, and so is where it was built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dicechess_training.contracts import prerank as contract
from dicechess_training.contracts.kcp13 import ContractError
from dicechess_training.prerank import export as exporter
from dicechess_training.prerank import pack as packer
from dicechess_training.prerank import roots as rooter
from dicechess_training.prerank.groups import KINDS, LICENSE_FILE, GroupsError


def _directory(raw: str) -> Path:
    resolved = Path(raw).resolve()
    if not resolved.is_dir():
        raise argparse.ArgumentTypeError("not a directory")
    return resolved


def _run_roots(args: argparse.Namespace) -> int:
    record = rooter.export(args.source, args.destination, args.limit)
    print(f"read {record['rows_read']:,} rows, {record['distinct_roots']:,} distinct roots")
    print(f"selected {record['roots']:,} roots from {record['games']:,} games")
    print(
        "  splits      "
        + "  ".join(f"{name} {count:,}" for name, count in record["splits"].items())
    )
    print(
        "  sides       " + "  ".join(f"{name} {count:,}" for name, count in record["sides"].items())
    )
    ply = record["ply"]
    print(
        f"  ply         min {ply['min']}  median {ply['median']}  "
        f"p95 {ply['p95']}  max {ply['max']}"
    )
    cost = rooter.projected_cost(record["roots"])
    print(
        f"  will cost   about {cost['candidates']:,} candidates, {cost['megabytes']:,} MB of JSON, "
        f"{cost['core_hours']} core-hours"
    )
    print(f"wrote {rooter.ROOTS_FILE} and {rooter.PROVENANCE_FILE}")
    print(f"source_sha256 {record['roots_sha256']}")
    return 0


def _run_pack(args: argparse.Namespace) -> int:
    manifest, summary = packer.pack(
        args.corpus,
        license_=args.license,
        license_file=args.license_file,
        version=args.version,
        kind=args.kind,
    )
    sizes = summary["candidates_per_group"]
    print(f"admitted {manifest['groups']:,} groups, {manifest['candidates']:,} candidates")
    print(f"  version     {manifest['version']}  kind {manifest['kind']}")
    print(f"  engine      {manifest['engine_version']}  schema {manifest['feature_schema']}")
    print(f"  teacher     {manifest['teacher']['id']}")
    print(
        "  splits      "
        + "  ".join(f"{name} {count:,}" for name, count in summary["splits"].items())
    )
    print(
        f"  per group   min {sizes['min']}  median {sizes['median']}  "
        f"p95 {sizes['p95']}  max {sizes['max']}"
    )
    print(
        f"  rankable    {summary['groups_above_shortlist']:,} groups larger than the shortlist "
        f"of 48, of {summary['groups']:,}"
    )
    print(
        f"  targets     median {summary['median_distinct_targets']} distinct per group, "
        f"{summary['groups_entirely_tied']:,} groups entirely tied"
    )
    if summary["roots_skipped"]:
        print("  roots skipped " + json.dumps(summary["roots_skipped"], sort_keys=True))
    return 0


def _run_export(args: argparse.Namespace) -> int:
    model = exporter.load_ranker(args.weights)
    provenance = dict(entry.split("=", 1) for entry in args.provenance)
    manifest = exporter.write_package(
        model,
        args.destination,
        model_id=args.model_id,
        engine_compatibility=args.engine_compatibility,
        provenance=provenance,
        engine_version=args.engine_version,
    )
    graph = Path(args.destination) / exporter.MODEL_FILE
    parity = exporter.probe_parity(model, graph, args.engine_version)
    print(f"exported {manifest['modelId']} as {manifest['modelRole']}")
    print(f"  manifest    {manifest['manifestVersion']}  schema {manifest['featureSchema']}")
    print(f"  engine      {manifest['engineCompatibility']}")
    print(f"  modelSha256 {manifest['modelSha256']}")
    print(f"  probe parity against the checkpoint: {parity:.3g}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dicechess_training.prerank", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    choose = commands.add_parser("roots", help="choose the roots a corpus will be built from")
    choose.add_argument("source", type=_directory, help="directory of schema-v0 Parquet shards")
    choose.add_argument("destination", type=Path, help="where to write the roots file")
    choose.add_argument(
        "--limit", type=int, default=None, help="how many roots to take (default: all of them)"
    )
    choose.set_defaults(handler=_run_roots)

    admit = commands.add_parser("pack", help="write and verify a generated corpus's manifest")
    admit.add_argument("corpus", type=_directory, help="directory holding groups.json")
    admit.add_argument("--license", required=True, help="the data terms, named")
    admit.add_argument(
        "--license-file",
        type=Path,
        default=None,
        help=f"the evidence for those terms; copied in as {LICENSE_FILE}",
    )
    admit.add_argument("--version", default=packer.DEFAULT_VERSION, help="dataset version")
    admit.add_argument("--kind", default=packer.DEFAULT_KIND, choices=KINDS, help="dataset origin")
    admit.set_defaults(handler=_run_pack)

    ship = commands.add_parser("export", help="export trained weights as a pre-ranker artifact")
    ship.add_argument("weights", type=Path, help="a checkpoint written by the training run")
    ship.add_argument("destination", type=Path, help="where to write model.onnx and manifest.json")
    ship.add_argument("--model-id", required=True, help="what to call this artifact")
    ship.add_argument(
        "--engine-compatibility",
        default=">=" + contract.DEFAULT_ENGINE_VERSION,
        help="the engine range this artifact is for",
    )
    ship.add_argument(
        "--engine-version",
        default=contract.DEFAULT_ENGINE_VERSION,
        help="the engine whose committed golden fixes the feature layout",
    )
    ship.add_argument(
        "--provenance",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="recorded in the manifest; repeatable",
    )
    ship.set_defaults(handler=_run_export)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (packer.PackError, rooter.RootsError, GroupsError, ContractError) as error:
        # These messages are written to say what, never where — see the module docstring.
        print(f"refused: {error}", file=sys.stderr)
        return 1
    except OSError:
        # A filesystem failure carries the path it failed on, and this command handles corpora
        # whose location is itself private. The operator can see their own permissions and disk;
        # what they must not get is a traceback naming a directory in a log or a pasted report.
        print("refused: a file could not be read or written", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
