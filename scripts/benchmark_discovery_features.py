"""Measure real historical discovery cutoffs without running model selection."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ima.feature_discovery_specs import DiscoverySpec
from ima.feature_program import synthesize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cutoffs",type=int,default=10000)
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    frame=pd.read_csv(args.dataset,low_memory=False)
    spec=DiscoverySpec(entities=("horse","jockey","trainer"),measurements=("speed_mps","beaten_lengths","carried_weight"),aggregates=("count","mean","std","min","max"),windows_days=(90,365,None),domain_history=True,race_relative=True,max_definitions=500)
    args.output.mkdir(parents=True,exist_ok=True)
    sample=np.unique(np.linspace(0,len(frame)-1,min(args.cutoffs,len(frame)),dtype=int))
    reports=[]
    with threadpool_limits(limits=1):
        for name,positions in (("representative",sample),("full",None)) if args.full else (("representative",sample),):
            started=time.monotonic()
            matrix,manifest=synthesize(frame,spec,cutoff_positions=positions)
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=="darwin" else 1024**2)
            report={"name":name,"history_rows":len(frame),"cutoffs":len(matrix),"features":len(matrix.columns),"wall_seconds":time.monotonic()-started,"process_peak_rss_gib":peak,"matrix_memory_gib":matrix.memory_usage(deep=True).sum()/1024**3,"spec":spec.model_dump(mode="json"),"engine_version":manifest["engine_version"],"scored_labels_used":False}
            reports.append(report)
            (args.output/f"{name}.json").write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
    (args.output/"report.json").write_text(json.dumps(reports,indent=2))


if __name__=="__main__": main()
