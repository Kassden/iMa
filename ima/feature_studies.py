"""Matched race-level evidence, never a universal cross-target score."""
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def race_objectives(predictions, target):
    rows=[]
    for (fold,race), group in predictions.groupby(["fold_id","race_id"],sort=True):
        if target=="win_probability":
            value=-float(group.target_win.to_numpy()@np.log(np.clip(group.model_probability,1e-12,1)))
        elif target=="ranking_strength":
            value=-float(ndcg_score([group.label.to_numpy()],[group.prediction.to_numpy()],k=3))
        elif target=="placing_top_k":
            value=float(np.mean((group.label-group.prediction)**2))
        else:
            value=float(np.mean(np.abs(group.label-group.prediction)))
        rows.append({"fold_id":fold,"race_id":race,"date":group.date.iloc[0],"objective":value,"runner_keys":tuple(sorted(group.horse_no.astype(str)))})
    return pd.DataFrame(rows)


def paired_feature_report(candidate_path, comparator_path, target, seed=17):
    candidate=race_objectives(pd.read_csv(candidate_path),target)
    comparator=race_objectives(pd.read_csv(comparator_path),target)
    keys=["fold_id","race_id"]
    paired=candidate.merge(comparator,on=keys,suffixes=("_candidate","_comparator"),validate="one_to_one")
    if not len(paired) or len(paired)!=len(candidate) or len(paired)!=len(comparator) or not paired.runner_keys_candidate.eq(paired.runner_keys_comparator).all():
        raise ValueError("Noncomparable scored race/runner populations")
    delta=(paired.objective_candidate-paired.objective_comparator).to_numpy()
    rng=np.random.default_rng(seed)
    draws=np.concatenate([rng.choice(delta,size=(100,len(delta)),replace=True).mean(axis=1) for _ in range(20)])
    low,high=np.quantile(draws,[.025,.975])
    meetings=paired.assign(delta=delta).groupby("date_candidate").delta.mean().to_numpy()
    blockdraw=np.concatenate([rng.choice(meetings,size=(100,len(meetings)),replace=True).mean(axis=1) for _ in range(20)])
    blocklow,blockhigh=np.quantile(blockdraw,[.025,.975])
    return {"target_kind":target,"races":len(paired),"candidate_mean":float(paired.objective_candidate.mean()),"comparator_mean":float(paired.objective_comparator.mean()),"paired_delta":float(delta.mean()),"race_bootstrap_ci95":[float(low),float(high)],"meeting_equal_weight_sensitivity_ci95":[float(blocklow),float(blockhigh)],"verdict":"keep" if high<0 and blockhigh<0 else "reject" if low>0 and blocklow>0 else "inconclusive","caveat":"Exploratory development comparison after adaptive selection; not a confirmation or profit claim"}
