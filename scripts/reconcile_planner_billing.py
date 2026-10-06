"""Run bounded OpenRouter billing recovery for an existing campaign."""

import argparse
import json
import os

from ima.openrouter_billing import reconcile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("OPENROUTER_API_KEY is required")
    report = reconcile(args.campaign, key, revision=args.revision)
    print(json.dumps(report))
    return 1 if report["unresolved"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
