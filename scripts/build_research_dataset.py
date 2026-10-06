"""Queue/build an immutable official research candidate without campaign promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.dataset_registry import DatasetRegistry
from ima.dataset_specs import DatasetRequest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="DatasetRequest JSON")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True, help="Acquisition/snapshot manifest whose identity is in request")
    parser.add_argument("--submit-only", action="store_true")
    args = parser.parse_args(argv)
    request = DatasetRequest.model_validate_json(args.request.read_text())
    registry = DatasetRegistry(args.registry)
    state = registry.submit(request)
    if args.submit_only:
        print(json.dumps(state, indent=2))
        return 0
    manifest = registry.build(request.request_id, source_snapshot=args.source_snapshot, raw_manifest=args.raw_manifest)
    print(json.dumps({key: manifest[key] for key in ("dataset_id", "status", "request_id", "rows", "races", "features_path", "protocol_path", "comparison_contract_id", "event_policy_id", "validation", "unsupported_requirements")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
