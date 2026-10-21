"""Inspect, explicitly provision bounded control state, or plan cache retirement."""
from __future__ import annotations
import argparse
from dataclasses import asdict
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
    control_init=commands.add_parser('control-init',help='Explicit healthy-storage bootstrap; does not authorize restart')
    control_init.add_argument('--authority',type=Path,required=True)
    control_init.add_argument('--policy',type=Path,required=True)
    control_init.add_argument('--policy-sha256',required=True)
    control_status=commands.add_parser('control-status',help='Read bounded control state; may recover its rollback journal')
    control_status.add_argument('--authority',type=Path,required=True)
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
    elif args.command in {'control-init','control-status'}:
        from ima.research_control import ControlIntentStore,ControlPolicy
        if args.command=='control-init':
            with args.policy.open('rb') as stream:encoded=stream.read(65537)
            if len(encoded)>65536 or hashlib.sha256(encoded).hexdigest()!=args.policy_sha256:
                raise ValueError('Explicit control policy digest or size mismatch')
            try:policy=ControlPolicy(**json.loads(encoded))
            except (TypeError,ValueError) as exc:raise ValueError('Invalid explicit control policy') from exc
            authority=StorageAuthority(args.authority)
            store=ControlIntentStore.bootstrap(authority,policy)
        else:
            if not args.authority.is_dir():raise ValueError('Existing authority required; no implicit control bootstrap')
            authority=StorageAuthority(args.authority)
            store=ControlIntentStore(authority)
        pending=[json.loads(raw) for raw in store.pending()]
        result={'mode':'explicit_healthy_control_bootstrap' if args.command=='control-init' else 'bounded_control_readback',
            'control_root':str(store.root),'policy':asdict(store.policy),
            'policy_identity_sha256':hashlib.sha256((store.root/'policy.json').read_bytes()).hexdigest(),
            'normal_headroom_bytes':authority.headroom_bytes,'allocated_bytes':store.allocated_bytes(),
            'pending_count':len(pending),'pending_intents':[
                {key:row[key] for key in ('campaign','destination','role','sequence','state','payload_hash')}
                for row in pending],
            'may_recover_bounded_journal':True,'rollout_qualified':False,'normal_restart_authorized':False}
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
