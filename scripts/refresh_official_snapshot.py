"""Build and independently verify changed official inputs outside the optimizer."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from filelock import FileLock

from scrapper.official_corpus import PARSER_VERSION
from scripts.build_official_dataset import build
from scripts.verify_official_snapshot import verify


def input_digest(archive, profiles, corpora):
    digest = hashlib.sha256()
    sources = [(archive, ("raw/**/*.gz", "normalized/**/*.json")),
               (profiles, ("raw/**/*.gz", "normalized/**/*.json"))]
    sources += [(root, ("raw/**/*.gz", "documents/*.json")) for root in corpora]
    for root, patterns in sources:
        if not root.is_dir():
            raise ValueError(f"Missing official input directory: {root}")
        digest.update(json.dumps(str(root.resolve())).encode())
        for path in sorted({path for pattern in patterns for path in root.glob(pattern)}):
            content = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    content.update(block)
            digest.update(json.dumps([str(path.relative_to(root)), content.hexdigest()]).encode())
    for name in ("scripts/build_official_dataset.py", "scripts/verify_official_snapshot.py",
                 "scripts/refresh_official_snapshot.py", "scrapper/official_corpus.py",
                 "scrapper/historical/results.py", "scrapper/horse_pages.py",
                 "ima/rich_features.py", "ima/data.py", "ima/feature_sets.py"):
        digest.update(json.dumps([name, hashlib.sha256(Path(name).read_bytes()).hexdigest()]).encode())
    try:
        pdf_version = version("pypdf")
    except PackageNotFoundError:
        pdf_version = None
    digest.update(json.dumps({"python": sys.version, "dependencies": {
        name: version(name) for name in ("numpy", "pandas", "pyarrow", "scrapy", "parsel", "lxml", "filelock")}}, sort_keys=True).encode())
    digest.update(json.dumps({"pypdf": pdf_version}).encode())
    return digest.hexdigest()


def refresh(archive, profiles, corpora, output, baseline_runners=None, baseline_manifest=None):
    if bool(baseline_runners) != bool(baseline_manifest):
        raise ValueError("Baseline runners and manifest must be supplied together")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "refresh.lock", timeout=0):
        fingerprint = input_digest(archive, profiles, corpora)
        if baseline_runners:
            fingerprint = hashlib.sha256(json.dumps([fingerprint,
                hashlib.sha256(baseline_runners.read_bytes()).hexdigest(),
                hashlib.sha256(baseline_manifest.read_bytes()).hexdigest()]).encode()).hexdigest()
        receipt_path = output / "latest_verified.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        base = None
        if previous:
            base = Path(previous["snapshot"])
            if not base.resolve().is_relative_to(output.resolve()):
                raise ValueError("Previous snapshot lies outside acquisition output")
            verify(base)
            if previous["input_digest"] == fingerprint:
                if baseline_runners:
                    verify(base, baseline_runners, baseline_manifest)
                return previous | {"action": "unchanged_verified_inputs"}
            if previous["parser_version"] != PARSER_VERSION:
                base = None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot = output / f"snapshot-{stamp}-{uuid4().hex[:8]}"
        build(SimpleNamespace(archive=archive, profiles=profiles, corpus=corpora,
                              base_snapshot=base, output=snapshot))
        report = verify(snapshot, baseline_runners, baseline_manifest)
        report_path = snapshot.with_suffix(".readback.json")
        report_path.write_text(json.dumps(report, indent=2))
        receipt = {"snapshot": str(snapshot.resolve()), "parser_version": PARSER_VERSION,
                   "input_digest": fingerprint, "verified_at": datetime.now(timezone.utc).isoformat(),
                   "readback": str(report_path.resolve()), "races": report["races"],
                   "runners": report["runners"], "policy": "acquisition only; no optimizer promotion"}
        # This is a rebuild trigger, not the snapshot identity: acquisition can advance during a build.
        receipt["input_digest_scope"] = "inputs at job start; immutable dataset identity is its manifest and lineage"
        if baseline_runners:
            receipt.update(baseline_runners=str(baseline_runners.resolve()),
                           baseline_manifest=str(baseline_manifest.resolve()),
                           baseline_runner_keys_retained=report["baseline_runner_keys_retained"])
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2))
        temporary.replace(receipt_path)
        return receipt | {"action": "built_and_verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-runners", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(refresh(args.archive, args.profiles, args.corpus, args.output,
                             args.baseline_runners, args.baseline_manifest), indent=2))


if __name__ == "__main__":
    main()
