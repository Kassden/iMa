"""Run an independent, bounded feature-search worker against a verified registry."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-features", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--main-campaign", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--code-revision", default="local")
    parser.add_argument("--environment-hash", default="local")
    parser.add_argument("--refill-size", type=int, default=32)
    parser.add_argument("--native-threads", type=int, default=2)
    parser.add_argument("--fit-timeout", type=float, default=900)
    parser.add_argument("--disk-reserve-bytes", type=int, default=10_000_000_000)
    parser.add_argument("command", choices=["run", "once", "refill", "status"])
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    if args.command == "status":
        path = args.root/"status.json"
        print(path.read_text() if path.is_file() else json.dumps({"operation": "not_started"}))
        return 0
    if not args.parent_features or not args.registry:
        parser.error("--parent-features and --registry are required for feature work")
    from ima.feature_exploration import ExplorationConfig, FeatureExplorer
    explorer = FeatureExplorer(ExplorationConfig(root=args.root, parent_features=args.parent_features,
        registry=args.registry, main_campaign=args.main_campaign, tracking_uri=args.tracking_uri,
        code_revision=args.code_revision, environment_hash=args.environment_hash,
        refill_size=args.refill_size, native_threads=args.native_threads,
        fit_timeout=args.fit_timeout, disk_reserve_bytes=args.disk_reserve_bytes))
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive; omit for continuous operation")
    if args.command == "refill":
        from ima.feature_exploration import exploration_lock
        with exploration_lock(args.root):
            explorer.refill()
            result = explorer.status("queued")
    else:
        result = explorer.run(max_steps=1 if args.command == "once" else args.max_steps)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


# The sealed overlay adds only new queue modules. Scientific consumers remain
# pinned to the existing campaign's implementation, including spawned workers.
if os.environ.get("IMA_FEATURE_SOURCE_ROOT"):
    import hashlib
    from importlib.metadata import version
    import ima
    stage = Path(__file__).resolve().parents[1]
    source = Path(os.environ["IMA_FEATURE_SOURCE_ROOT"]).resolve()
    for directory, name in ((stage, "feature-exploration-manifest.json"), (source, "lifecycle-manifest.json")):
        manifest = json.loads((directory/name).read_text())
        if name == "feature-exploration-manifest.json":
            for package, expected in manifest["dependencies"].items():
                if version(package) != expected:
                    raise RuntimeError("Feature dependency changed: "+package)
        for relative, digest in manifest["files"].items():
            path = directory/relative
            if path.is_symlink() or not path.resolve().is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Sealed feature implementation checksum mismatch: "+relative)
    paths = [str(stage/"ima"), str(source/"ima")]
    ima.__path__[:] = list(dict.fromkeys(paths + list(ima.__path__)))

if __name__ == "__main__":
    raise SystemExit(main())
