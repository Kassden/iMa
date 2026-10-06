"""Deterministic objective/centering benchmark; never trains or promotes a model."""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from ima.performance_probit import GaussianRaceProbit


def benchmark(races=900, runners=14, features=24, repeats=2, threads=1):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(races*runners, features))
    codes = np.repeat(np.arange(races), runners)
    groups = list(np.arange(len(x)).reshape(races, runners))
    winners = rng.integers(runners, size=races).tolist()
    model = GaussianRaceProbit(tuple(f"f{i}" for i in range(features)),
                              heteroscedastic=True, l2=1., scale_l2=1., quadrature_order=32)
    parameters = rng.normal(scale=.01, size=2*features)
    samples = []
    profiler = cProfile.Profile()
    with threadpool_limits(limits=threads):
        model._objective_and_gradient(parameters, x, codes, groups, winners)
        profiler.enable()
        for _ in range(repeats):
            started = perf_counter()
            value, gradient = model._objective_and_gradient(parameters, x, codes, groups, winners)
            samples.append(perf_counter()-started)
        profiler.disable()
        started = perf_counter()
        scales = model._scales(x, parameters[features:], codes)
        scale_seconds = perf_counter()-started
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(15)
    return {"seed":42, "races":races, "runners":runners, "features":features,
            "threads":threads, "seconds":samples, "median_seconds":float(np.median(samples)),
            "scale_seconds":scale_seconds, "objective":float(value),
            "gradient":gradient.tolist(), "scale_sum":float(scales.sum()),
            "profile":output.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("races",900),("runners",14),("features",24),("repeats",2),("threads",1)):
        parser.add_argument("--"+name,type=int,default=default)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if any(getattr(args,name)<1 for name in ("races","runners","features","repeats","threads")):
        parser.error("Shapes, repetitions and threads must be positive")
    result = benchmark(args.races,args.runners,args.features,args.repeats,args.threads)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
