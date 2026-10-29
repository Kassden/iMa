"""Run research confirmation commands.

The legacy ``lock``/``evaluate`` commands consume frozen prediction CSVs and
separate outcome labels. The ``meeting-*`` commands use hash-pinned native
research packages, an official source-bound future calendar, and sealed package
replay; they do not accept operator-supplied probability CSVs. Neither workflow
provides automatic promotion evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.research_confirmation import OBJECTIVE, ExposureLedger, create_lock, evaluate_lock


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    lock = commands.add_parser("lock", help="Freeze identities, hashes, contract, and later population")
    lock.add_argument("--candidate", type=Path, required=True)
    lock.add_argument("--baseline", type=Path, required=True)
    lock.add_argument("--candidate-identity", type=_json_object, required=True)
    lock.add_argument("--baseline-identity", type=_json_object, required=True)
    lock.add_argument("--selection-cutoff", required=True)
    lock.add_argument("--prospective", action="store_true",
                      help="Lock label-free forecasts for strictly future UTC race dates")
    lock.add_argument("--objective", choices=[OBJECTIVE], default=OBJECTIVE)
    lock.add_argument("--confidence-level", type=float, default=0.95)
    lock.add_argument("--bootstrap-resamples", type=int, default=20_000)
    lock.add_argument("--random-seed", type=int, default=20261008)
    lock.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Evaluate an existing immutable lock")
    evaluate.add_argument("--lock", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path,
                          help="Separate outcomes CSV; required for prospective locks")
    exposure=commands.add_parser('exposure',help='Record known selection or manual label exposure')
    exposure.add_argument('--ledger',type=Path,required=True)
    exposure.add_argument('--first-date',required=True)
    exposure.add_argument('--last-date',required=True)
    exposure.add_argument('--campaign',required=True)
    exposure.add_argument('--purpose',required=True)
    exposure.add_argument('--evidence',type=Path,required=True)
    seal=commands.add_parser('seal',help='Freeze forecasts and an explicit source-bound exposure inventory')
    seal.add_argument('--manifest',type=Path,required=True)
    seal.add_argument('--lock',type=Path,required=True)
    seal.add_argument('--output',type=Path,required=True)
    study=commands.add_parser('evaluate-study',help='Check a study seal and evaluate only after its horizon')
    study.add_argument('--seal',type=Path,required=True)
    study.add_argument('--output',type=Path,required=True)
    study.add_argument('--labels',type=Path)
    meeting_lock=commands.add_parser('meeting-lock',help='Verify native packages, official source calendar, exposure and power before locking a fixed horizon')
    meeting_lock.add_argument('--manifest',type=Path,required=True,
                              help='JSON manifest includes calendar/source-readback paths and a paired development CSV keyed to both package hashes')
    meeting_lock.add_argument('--output',type=Path,required=True)
    meeting_seal=commands.add_parser('meeting-seal',help='Replay locked native packages on pinned feature inputs and seal predictions before race deadlines')
    meeting_seal.add_argument('--lock',type=Path,required=True)
    meeting_seal.add_argument('--meeting-date',required=True)
    meeting_seal.add_argument('--official-card',type=Path,required=True)
    meeting_seal.add_argument('--runners',type=Path,required=True)
    meeting_seal.add_argument('--inputs',type=Path,required=True,
                              help='Label-free receipt for a hash-pinned Parquet feature frame')
    meeting_seal.add_argument('--output-root',type=Path,required=True)
    meeting_status=commands.add_parser('meeting-status',help='Report label-free forecast collection coverage and operational misses')
    meeting_status.add_argument('--lock',type=Path,required=True)
    meeting_status.add_argument('--versions-root',type=Path,required=True)
    meeting_evaluate=commands.add_parser('meeting-evaluate',help='Evaluate the one fixed horizon after all locked outcomes are available')
    meeting_evaluate.add_argument('--lock',type=Path,required=True)
    meeting_evaluate.add_argument('--versions-root',type=Path,required=True)
    meeting_evaluate.add_argument('--outcomes',type=Path,required=True)
    meeting_evaluate.add_argument('--output',type=Path,required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "lock":
            path = create_lock(
                args.candidate, args.baseline, args.output,
                candidate_identity=args.candidate_identity,
                baseline_identity=args.baseline_identity,
                selection_cutoff=args.selection_cutoff,
                prospective=args.prospective,
                objective=args.objective,
                confidence_level=args.confidence_level,
                bootstrap_resamples=args.bootstrap_resamples,
                random_seed=args.random_seed,
            )
        elif args.command=='evaluate':
            path = evaluate_lock(args.lock, args.output, labels_path=args.labels)
        elif args.command=='exposure':
            receipt=ExposureLedger(args.ledger).record(first_date=args.first_date,last_date=args.last_date,
                campaign=args.campaign,purpose=args.purpose,evidence=args.evidence)
            print(json.dumps({'command':args.command,'receipt_id':receipt,'ledger':str(args.ledger)}))
            return 0
        elif args.command=='seal':
            from ima.research_study_seal import create_study_seal
            path=create_study_seal(json.loads(args.manifest.read_text()),args.lock,args.output)
        elif args.command=='evaluate-study':
            from ima.research_study_seal import evaluate_study_seal
            result=evaluate_study_seal(args.seal,args.output,labels_path=args.labels)
            print(json.dumps({'command':args.command,**result},sort_keys=True))
            return 0
        elif args.command=='meeting-lock':
            from ima.research_meeting_study import create_meeting_study
            path=create_meeting_study(json.loads(args.manifest.read_text()),args.output)
        elif args.command=='meeting-seal':
            from ima.research_meeting_study import seal_meeting_version
            path=seal_meeting_version(args.lock,args.meeting_date,args.official_card,args.runners,args.inputs,
                                      args.output_root)
        elif args.command=='meeting-status':
            from ima.research_meeting_study import collection_status
            print(json.dumps({'command':args.command,**collection_status(args.lock,args.versions_root)},sort_keys=True))
            return 0
        elif args.command=='meeting-evaluate':
            from ima.research_meeting_study import evaluate_fixed_horizon
            path=evaluate_fixed_horizon(args.lock,args.versions_root,args.outcomes,args.output)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"command": args.command, "artifact": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
