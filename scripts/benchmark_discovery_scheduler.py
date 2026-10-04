"""Isolated pool versus Ray Tune queue overhead; not a predictive benchmark."""
import argparse
import json
import multiprocessing
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path


def fit_task(rows, iterations):
    import numpy as np
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_limits
    from ima.research_resources import JobMonitor
    with JobMonitor() as monitor, threadpool_limits(limits=1):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(rows,64))
        y = rng.normal(size=rows)
        for _ in range(iterations):
            model = Ridge().fit(x,y)
        checksum = float(model.predict(x[:8]).sum())
    return {"checksum":checksum,"measurement":monitor.report()}


def resource_benchmark(args):
    from ima.research_resources import JobWorkload, estimate_job, JobEstimator
    from ima.research_scheduler import ResourceAdmission
    requests = [("long",10000,30),("short",2000,1),("medium",4000,4),("backfill",1000,1)]
    estimator = JobEstimator(args.output/"job-samples.json")
    runs = []
    for repeat in range(args.repeats):
        admission = ResourceAdmission(2,2,2)
        work = {key:JobWorkload(stage="fit",family="ridge",rows=rows,generated_features=64,selected_features=64,search_settings={"iterations":iterations},implementation_revision="scheduler-benchmark-v1",dependency_versions={"benchmark":"1"}) for key,rows,iterations in requests}
        pending = [(key,estimate_job(work[key],cold_private_bytes=256*1024**2)) for key,_,_ in requests]
        dimensions = {key:(rows,iterations) for key,rows,iterations in requests}
        running,events,finished = {},[],[]
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn"),max_tasks_per_child=1) as executor:
            while pending or running:
                while True:
                    candidate = admission.peek_feasible(pending)
                    if candidate is None:
                        break
                    key,request = candidate
                    admission.reserve(key,request)
                    pending.remove(candidate)
                    future = executor.submit(fit_task,*dimensions[key])
                    running[future] = key
                    events.append({"event":"dispatch","key":key,"seconds":time.monotonic()-started,"resources":admission.snapshot()})
                if not running:
                    raise RuntimeError(f"No feasible benchmark work: {admission.last_blockers}")
                done,_ = wait(running,return_when=FIRST_COMPLETED)
                for future in done:
                    key = running.pop(future)
                    actual = future.result()
                    admission.release(key)
                    estimator.record(work[key],actual["measurement"])
                    finished.append({"key":key,**actual})
                    events.append({"event":"complete","key":key,"seconds":time.monotonic()-started})
        seconds = time.monotonic()-started
        runs.append({"repeat":repeat,"wall_seconds":seconds,"trials_per_hour":len(finished)*3600/seconds,"events":events,"finished":finished})
    report = {"scope":"Local measured Ridge workloads with unequal durations; no production-throughput claim","concurrency":2,"threads_per_fit":1,"runs":runs,"estimator_samples":len(estimator.samples)}
    (args.output/"resource-benchmark.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"concurrency":2,"runs":len(runs),"wall_seconds":[r["wall_seconds"] for r in runs],"completed":sum(len(r["finished"]) for r in runs)}))


def task(seconds):
    started=time.monotonic()
    time.sleep(seconds)
    return {"requested_seconds":seconds,"work_seconds":time.monotonic()-started}


def ray_task(config):
    from ray import tune
    tune.report(task(config["seconds"]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resource",action="store_true")
    parser.add_argument("--repeats",type=int,default=2)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resource:
        if args.repeats < 1:
            parser.error("Positive repetition count required")
        return resource_benchmark(args)
    requests=[.4,.05,.1,.08]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as executor:
        finished=[future.result() for future in as_completed([executor.submit(task,t) for t in requests])]
    pool={"wall_seconds":time.monotonic()-started,"completed":len(finished),"completion_order":[r["requested_seconds"] for r in finished]}
    (args.output/"pool.json").write_text(json.dumps(pool,indent=2))
    started=time.monotonic()
    ray_report={}
    runtime=tempfile.TemporaryDirectory(prefix="ima-ray-")
    try:
        import ray
        from ray import tune
        ray.init(num_cpus=2,num_gpus=0,include_dashboard=False,object_store_memory=100*1024**2,_node_ip_address="127.0.0.1",_temp_dir=runtime.name)
        analysis=tune.run(ray_task,config={"seconds":tune.grid_search(requests)},resources_per_trial={"cpu":1},max_concurrent_trials=2,storage_path=str((args.output/"ray-results").resolve()),verbose=0)
        ray_report={"wall_seconds":time.monotonic()-started,"completed":len(analysis.trials),"version":ray.__version__}
    except Exception as exc:
        ray_report={"status":"unsupported_in_isolated_canary","error":f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            import ray
            ray.shutdown()
        except ImportError:
            pass
        runtime.cleanup()
    result={"pool":pool,"ray_tune":ray_report,"selected":"persistent_spawn_pool","rationale":"Single audited host; existing single-owner ledger/Optuna controller and pool need no distributed scheduler. Ray is optional only for a demonstrated missing capability.","scope":"Queue initialization and replenishment overhead on deterministic sleep tasks, not model-training throughput"}
    (args.output/"comparison.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
