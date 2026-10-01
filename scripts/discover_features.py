"""Terminal-first deterministic feature discovery and replay."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ima.feature_discovery_specs import DiscoverySpec
from ima.feature_program import materialize, synthesize
from ima.feature_screening import DiscoverySelection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("enumerate","materialize","screen","report","replay"))
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--train-races",type=Path)
    parser.add_argument("--target",default="win_probability")
    parser.add_argument("--label",default="target_win")
    args=parser.parse_args()
    spec=DiscoverySpec.model_validate(json.loads(args.spec.read_text())) if args.spec else DiscoverySpec()
    frame=pd.read_parquet(args.dataset) if args.dataset.suffix==".parquet" else pd.read_csv(args.dataset,low_memory=False)
    digest=hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.command=="enumerate":
        _,manifest=synthesize(frame,spec,enumerate_only=True)
        print(json.dumps({"discovery_id":manifest["discovery_id"],"candidate_count":len(manifest["catalog"]),"catalog":manifest["catalog"],"deferred":manifest["deferred"]},indent=2))
        return
    enriched,manifest=materialize(frame,spec,digest,args.output)
    if args.command=="screen":
        if not args.train_races: parser.error("screen requires explicit --train-races; no full-data selection")
        races=set(json.loads(args.train_races.read_text()))
        train=enriched[enriched.race_id.astype(str).isin(races)]
        if train.empty: parser.error("no matching training races")
        result=DiscoverySelection.fit(train,spec,args.target,args.label).report
    elif args.command=="replay":
        repeated,_=synthesize(frame,spec)
        pd.testing.assert_frame_equal(enriched[list(repeated)].reset_index(drop=True),repeated.reset_index(drop=True))
        result={"replay":"passed","matrix_id":manifest["matrix_id"]}
    else:
        result={"discovery_id":manifest["discovery_id"],"matrix_id":manifest["matrix_id"],"candidate_count":len(manifest["catalog"]),"deferred_count":manifest["deferred_count"],"catalog":manifest["catalog"],"coverage":manifest["coverage"]}
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__": main()
