"""Read-only research-object lifecycle planner and explicit copy/restore CLI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ima.research_object_lifecycle import (
    ProtectedRoot, archive_object, plan_retention, restore_object,
)
from ima.research_storage import StorageAuthority


def _root(value):
    kind, sep, path = value.partition("=")
    if not sep or not kind or not path:
        raise ValueError("Protected roots use KIND=PATH")
    return ProtectedRoot(kind, Path(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Read-only dry run; does not authorize deletion")
    plan.add_argument("--object-root", type=Path, required=True)
    plan.add_argument("--snapshot-root", type=Path, action="append", default=[])
    plan.add_argument("--dataset-registry", type=Path, action="append", default=[])
    plan.add_argument("--protected-root", action="append", type=_root, required=True,
                      help="Repeat for mlflow, hypothesis, confirmation, pending, models, raw, unique_history")
    plan.add_argument("--now", type=float)
    plan.add_argument("--grace-seconds", type=int, default=7 * 86400)

    for name in ("archive", "restore"):
        command = commands.add_parser(name)
        command.add_argument("--object-root", type=Path, required=True)
        command.add_argument("--reference", type=Path, required=True,
                             help="JSON object reference with sha256, size, and format")
        command.add_argument("--authority", type=Path, required=True,
                             help="Existing StorageAuthority root")
    archive = commands.choices["archive"]
    archive.add_argument("--destination", type=Path, required=True)
    restore = commands.choices["restore"]
    restore.add_argument("--archive", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan_retention(object_root=args.object_root, snapshot_roots=args.snapshot_root,
                                dataset_registry_roots=args.dataset_registry,
                                protected_roots=args.protected_root, now=args.now,
                                grace_seconds=args.grace_seconds)
    else:
        if not args.authority.is_dir():
            raise ValueError("Existing StorageAuthority root required; CLI will not initialize one")
        reference = json.loads(args.reference.read_text())
        authority = StorageAuthority(args.authority)
        if args.command == "archive":
            result = archive_object(object_root=args.object_root, reference=reference,
                                    destination=args.destination, authority=authority)
        else:
            result = restore_object(archive_path=args.archive, object_root=args.object_root,
                                   reference=reference, authority=authority)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "held", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
