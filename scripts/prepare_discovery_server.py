"""Prepare isolated imaopt v5 inputs and canary, without changing running services."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("/home/imaopt/research-v2"))
    parser.add_argument("--canary-name",default="agentic_v5_canary_a")
    args=parser.parse_args()
    if Path.home()!=Path("/home/imaopt"):
        raise SystemExit("Run as dedicated imaopt server user, not another service identity")
    root=args.root.resolve()
    if not root.is_relative_to(Path.home()):
        raise SystemExit("Root must remain under imaopt home")
    source=root/"campaigns/agentic_v4_features/inputs"
    live=root/"campaigns/agentic_v5_discovery"
    for folder in ("inputs","ops"):
        (live/folder).mkdir(parents=True,exist_ok=True)
    target=live/"inputs/rich-history-v5.csv.gz"
    if not target.exists():
        shutil.copy2(source/"rich-history-v4.csv.gz",target)
    if not (live/"inputs/protocol.json").exists():
        shutil.copy2(source/"protocol.json",live/"inputs/protocol.json")
    config=json.loads((Path(__file__).resolve().parents[1]/"config/agentic_v5_discovery.json").read_text())
    (live/"ops/openrouter-config.json").write_text(json.dumps(config,indent=2))
    canary=root/"campaigns"/args.canary_name
    if canary.exists():
        raise SystemExit("Canary exists; use a new identity, never overwrite evidence")
    (canary/"inputs").mkdir(parents=True)
    (canary/"ops").mkdir()
    frame=pd.read_csv(target,low_memory=False)
    ordered=frame[["race_id","date","race_no"]].drop_duplicates("race_id").sort_values(["date","race_no"],kind="stable")
    races=set(ordered.race_id.iloc[:500])
    subset=frame[frame.race_id.isin(races)]
    dataset=canary/"inputs/history.csv.gz"
    subset.to_csv(dataset,index=False,compression="gzip")
    protocol=canary/"inputs/protocol.json"
    protocol.write_text(json.dumps({"min_train_races":200,"calibration_races":50,"score_races":50,"max_folds":2}))
    config.update(max_trials=20,max_trials_per_decision=20,max_concurrent_trials=2,cpu_thread_budget=2,ram_budget_gib=8,planning_checkpoint_seconds=300,replan_every_terminal_trials=8,max_total_cost_usd=1,dataset_path=str(dataset),protocol_path=str(protocol))
    (canary/"ops/openrouter-config.json").write_text(json.dumps(config,indent=2))
    print(json.dumps({"main_dataset_rows":len(frame),"main_races":frame.race_id.nunique(),"canary_rows":len(subset),"canary_races":len(races),"campaign":str(canary)}))


if __name__=="__main__": main()
