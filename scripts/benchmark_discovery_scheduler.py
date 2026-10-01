"""Isolated pool versus Ray Tune queue overhead; not a predictive benchmark."""
import argparse
import json
import multiprocessing
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


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
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
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
