"""Inspect storage or plan exact own-user fold-cache retirement; fail closed."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from ima.research_storage import StorageAuthority, payload_identity, reclaim_payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    inspect=commands.add_parser('inspect')
    inspect.add_argument('--root',type=Path,required=True)
    status=commands.add_parser('status')
    status.add_argument('--authority',type=Path,required=True)
    plan=commands.add_parser('plan')
    plan.add_argument('--root',type=Path,required=True)
    plan.add_argument('--cache-root',type=Path,action='append',required=True)
    plan.add_argument('--output',type=Path,required=True)
    apply=commands.add_parser('apply')
    apply.add_argument('--manifest',type=Path,required=True)
    apply.add_argument('--manifest-sha256',required=True)
    apply.add_argument('--canary',type=Path,required=True)
    apply.add_argument('--authority',required=True)
    args=parser.parse_args(argv)
    if args.command=='inspect':
        root=args.root.absolute()
        if root.resolve()!=root or root.stat().st_uid!=os.getuid():
            raise ValueError('Inspect requires an own real root')
        s=os.statvfs(root)
        result={'uid':os.getuid(),'root':str(root),'available_bytes':s.f_bavail*s.f_frsize,
            'free_bytes':s.f_bfree*s.f_frsize,'available_inodes':s.f_favail,
            'mode':'read_only','reclaim_authorized':False}
    elif args.command=='status':
        result=StorageAuthority(args.authority).status()
    elif args.command=='plan':
        root=args.root.absolute(); records=[]
        for cache in args.cache_root:
            cache=cache.absolute()
            if not cache.is_relative_to(root) or cache.name!='fold-preparation-cache' or cache.is_symlink():
                raise ValueError('Cache root outside selected scope')
            for payload in sorted(cache.iterdir()):
                if payload.is_dir(): records.append(payload_identity(payload,root))
        result={'schema_version':1,'approved':False,'root':str(root),'candidates':records,
            'allocated_bytes':sum(r['allocated_bytes'] for r in records),
            'required_gates':['fresh real-package canary','same UID scope','no open mappings',
                'stable locks','producer quiescence','preserved rebuild inputs','explicit selected authority']}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    else:
        encoded=args.manifest.read_bytes()
        if hashlib.sha256(encoded).hexdigest()!=args.manifest_sha256:
            raise ValueError('Selected manifest digest changed')
        manifest=json.loads(encoded)
        if manifest.get('approved') is not True or manifest.get('authority')!=args.authority:
            raise ValueError('Manifest is not explicitly approved by this authority')
        canary=json.loads(args.canary.read_text())
        if not manifest.get('quiescence_verified') or not manifest.get('rebuild_inputs_verified'):
            raise ValueError('Producer/reference prerequisites not qualified')
        # Each receipt is emitted immediately; a partial failure cannot erase history.
        receipts=[]
        for record in manifest['candidates']:
            if record['root']!=manifest['root']: raise ValueError('Manifest scope mismatch')
            receipt=reclaim_payload(record,authority=args.authority,canary_receipt=canary)
            receipts.append(receipt); print(json.dumps({'receipt':receipt}),flush=True)
        result={'receipts':receipts,'mode':'selected_reclaim','candidates_completed':len(receipts)}
    print(json.dumps(result,indent=2),flush=True)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as exc:
        print(json.dumps({'status':'blocked','error':str(exc)}),file=sys.stderr)
        raise SystemExit(2)
