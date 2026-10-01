"""Exercise deterministic feature creation and the continuous feedback controller."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ima.optimizer import CampaignConfig
from ima.research_v5 import run_v5_campaign


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--planner",choices=("fixture","openrouter"),default="fixture")
    p.add_argument("--max-concurrent-trials",type=int,default=2)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--max-trials",type=int,default=20)
    p.add_argument("--model",default="deepseek/deepseek-v4.1-flash")
    p.add_argument("--mlflow-tracking-uri")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for day in range(80):
        for horse in range(6):
            result=(horse-day)%6+1
            rows.append(dict(race_id=f"r{day}",horse_no=horse+1,horse_id=f"h{horse}",jockey_id=f"j{horse%3}",trainer_id=f"t{horse%2}",date=(pd.Timestamp("2020-01-01")+pd.Timedelta(days=day*2)).isoformat(),race_no=1,finish_seconds=70+horse+np.sin(day),finish_time=70+horse+np.sin(day),distance=1200,lengths_raw=horse,actual_weight=120+horse,horse_age=3+horse%4,horse_rating=50+horse,declared_weight=1100+horse*20,draw=horse+1,race_class=3,surface=0,venue="ST",course="TURF",going="GOOD",field_size=6,result=result,target_win=int(result==1),market_probability=1/6,win_odds=5+horse))
    dataset=args.output/"fixture.csv"
    frame=pd.DataFrame(rows)
    frame["last_speed_ratio"]=frame.horse_no/10
    frame["prior_win_rate"]=.1+frame.horse_no/100
    frame["target_probability"]=frame.target_win
    frame.to_csv(dataset,index=False)
    protocol=args.output/"protocol.json"
    protocol.write_text(json.dumps({"min_train_races":30,"calibration_races":10,"score_races":10,"max_folds":2}))
    config=CampaignConfig(campaign_dir=args.output/"campaign",policy="agentic",research_policy="discovery_v5",planner_mode=args.planner,model=args.model,max_total_cost_usd=1,max_trials=args.max_trials,proposal_batch_size=args.max_trials,max_concurrent_trials=args.max_concurrent_trials,dataset_path=dataset,protocol_path=protocol,cpu_thread_budget=2,ram_budget_gib=8,planning_checkpoint_seconds=30,replan_every_terminal_trials=4,max_consecutive_failed_trials=3,mlflow_tracking_uri=args.mlflow_tracking_uri)
    result=run_v5_campaign(config)
    print(json.dumps({"mode":result["mode"],"ledger":result["ledger"],"cycles":result["cycles"],"campaign_dir":result["campaign_dir"]},indent=2,default=str))


if __name__=="__main__": main()
