"""Measure real historical discovery cutoffs without running model selection."""
import argparse
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ima.feature_discovery_specs import DiscoverySpec
from ima.feature_program import synthesize
from ima.research_resources import JobMonitor


def preparation_benchmark(frame, args):
    """Equal full-history populations, training-only median fit and actual Ridge replay."""
    from sklearn.linear_model import Ridge
    from sklearn import __version__ as sklearn_version
    import joblib
    from ima.feature_sets import NOTEBOOK_RICH_SCHEMA
    from ima.research_preparation import PreparationKey, PreparationCache, build_numeric_fold

    features = tuple(c for c in NOTEBOOK_RICH_SCHEMA.numeric if c in frame and pd.api.types.is_numeric_dtype(frame[c]))
    if not features or not {"date","race_id","horse_no","target_win"} <= set(frame):
        raise ValueError("Preparation benchmark requires the validated rich-history frame")
    frame = frame.sort_values(["date","race_id","horse_no"],kind="stable").reset_index(drop=True)
    discovery_spec = None
    candidate_count = len(features)
    if args.selection_budget is not None:
        from ima.feature_discovery_specs import DiscoverySpecV2, content_id
        budget = None if args.selection_budget == "all" else int(args.selection_budget)
        discovery_spec = DiscoverySpecV2(selection=args.selection_method,max_selected=budget)
        renamed = {name:"dfs_"+content_id({"registered_numeric":name}) for name in features}
        frame = frame.rename(columns=renamed)
        features = tuple(renamed.values())
    dates = pd.to_datetime(frame.date)
    unique = sorted(dates.unique())
    boundaries = unique[int(len(unique)*.7)],unique[int(len(unique)*.85)]
    columns = list(features)+["target_win","date","race_id"]
    populations = {"train":frame.loc[dates<boundaries[0],columns],"calibration":frame.loc[(dates>=boundaries[0]) & (dates<boundaries[1]),columns],"score":frame.loc[dates>=boundaries[1],columns]}
    runner_keys = frame.race_id.astype(str)+"/"+frame.horse_no.astype(str)
    row_keys = {name:tuple(runner_keys.loc[population.index]) for name,population in populations.items()}
    source = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    key = PreparationKey(source_content_hash=source,training_row_keys=row_keys["train"],calibration_row_keys=row_keys["calibration"],score_row_keys=row_keys["score"],target_labels_hash=hashlib.sha256(frame.target_win.to_numpy().tobytes()).hexdigest(),availability_policy={"owner":"validated-rich-history","cutoff":"strict"},feature_definitions=tuple({"name":name,"owner":"NOTEBOOK_RICH_SCHEMA"} for name in features),selection_settings=discovery_spec.model_dump(mode="json") if discovery_spec else {"benchmark":"all_registered_numeric"},fitted_transform_settings={"imputer":"median"},fold_dates=tuple(map(str,boundaries)),seed=17,implementation_revision=hashlib.sha256(Path("ima/research_preparation.py").read_bytes()).hexdigest(),dependency_versions={"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn_version,"joblib":joblib.__version__})
    stages = []
    first_prediction = None
    with tempfile.TemporaryDirectory(prefix="ima-preparation-",dir=args.output) as root:
        cache = PreparationCache(root)
        builder = lambda:build_numeric_fold(populations["train"],populations["calibration"],populations["score"],base_features=() if discovery_spec else features,label="target_win",row_keys=row_keys,discovery_spec=discovery_spec)
        with threadpool_limits(limits=1):
            for repeat in range(args.repeats):
                with JobMonitor() as preparation_monitor:
                    artifact = cache.prepare_fold(key,builder)
                with cache.pin(artifact), JobMonitor() as fit_monitor:
                    arrays = artifact.load_arrays()
                    model = Ridge(alpha=1).fit(arrays["train_x"],arrays["train_y"])
                    predictions = model.predict(arrays["score_x"])
                if first_prediction is None:
                    first_prediction = predictions
                np.testing.assert_allclose(predictions,first_prediction,rtol=1e-10,atol=1e-12)
                stages.append({"repeat":repeat,"cache_status":artifact.cache_status,"preparation":preparation_monitor.report(),"fit":fit_monitor.report(),"array_bytes":sum(a.nbytes for a in arrays.values()),"readonly":all(not a.flags.writeable for a in arrays.values()),"prediction_sha256":hashlib.sha256(predictions.tobytes()).hexdigest()})
        selection = artifact.load_fitted_state()["selection"]
        report = {"mode":"full_history_preparation","source_sha256":source,"rows":len(frame),"population_counts":{name:len(pop) for name,pop in populations.items()},"candidate_features":candidate_count,"numeric_features":len(artifact.manifest()["feature_names"]),"selection":selection.report if selection else None,"artifact_id":key.cache_id(),"cache":cache.stats,"stages":stages,"parity":True,"threads":1,"scope":"Local full-history numeric fold preparation and Ridge replay. No claim of production PSS/cgroup admission or feature-source correctness."}
    (args.output/"preparation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {"stages","selection"}}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cutoffs",type=int,default=10000)
    parser.add_argument("--full",action="store_true")
    parser.add_argument("--preparation",action="store_true")
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--selection-budget",help="Explicit positive integer or all; benchmark training-only registered numeric candidates")
    parser.add_argument("--selection-method",choices=["quality","mutual_information","embedded","sequential"],default="quality")
    args=parser.parse_args()
    frame=pd.read_csv(args.dataset,low_memory=False)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.preparation:
        if args.repeats < 2:
            parser.error("Preparation benchmark needs cold and warm repetitions")
        return preparation_benchmark(frame,args)
    spec=DiscoverySpec(entities=("horse","jockey","trainer"),measurements=("speed_mps","beaten_lengths","carried_weight"),aggregates=("count","mean","std","min","max"),windows_days=(90,365,None),domain_history=True,race_relative=True,max_definitions=500)
    args.output.mkdir(parents=True,exist_ok=True)
    sample=np.unique(np.linspace(0,len(frame)-1,min(args.cutoffs,len(frame)),dtype=int))
    reports=[]
    with threadpool_limits(limits=1):
        for name,positions in (("representative",sample),("full",None)) if args.full else (("representative",sample),):
            with JobMonitor() as monitor:
                matrix,manifest=synthesize(frame,spec,cutoff_positions=positions)
            report={"name":name,"history_rows":len(frame),"cutoffs":len(matrix),"features":len(matrix.columns),"measurement":monitor.report(),"matrix_memory_gib":matrix.memory_usage(deep=True).sum()/1024**3,"spec":spec.model_dump(mode="json"),"engine_version":manifest["engine_version"],"scored_labels_used":False}
            reports.append(report)
            (args.output/f"{name}.json").write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
    (args.output/"report.json").write_text(json.dumps(reports,indent=2))


if __name__=="__main__": main()
