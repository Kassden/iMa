#!/usr/bin/env python3
"""Audit frozen decisions without network access or campaign mutations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ima.proposal_audit import audit_decisions, render_markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "markdown"),
                        help="Default: JSON for .json output, Markdown otherwise")
    args = parser.parse_args(argv)
    try:
        if not args.decisions.is_dir():
            raise ValueError("--decisions must be a directory")
        paths = sorted(args.decisions.glob("D*.json"))
        if args.output.resolve() in {p.resolve() for p in paths + [args.evidence]}:
            raise ValueError("Output must not overwrite an input")
        sources, decisions = [], []
        for path in paths + [args.evidence]:
            content = path.read_bytes()
            sources.append(dict(path=str(path), sha256=hashlib.sha256(content).hexdigest()))
            value = json.loads(content)
            if path == args.evidence:
                evidence = value
            else:
                decisions.append(value)
        summary = audit_decisions(decisions, evidence)
        summary["source_files"] = sources
        fmt = args.format or ("json" if args.output.suffix.lower() == ".json" else "markdown")
        text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n" if fmt == "json" else (
            render_markdown(summary) + "\n## Source File SHA-256\n\n```json\n" + json.dumps(sources, indent=2) + "\n```\n")
        args.output.write_text(text, encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"audit error: {exc}\n")
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
