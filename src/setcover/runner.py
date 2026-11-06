import time
from pathlib import Path

import numpy as np

from setcover.io import read_instance
from setcover.registry import METHODS
from setcover.result import verify


def run_one(method: str, instance_path: str | Path, seed: int, time_limit: float, **params) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, available: {sorted(METHODS)}")
    inst = read_instance(instance_path)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    res = METHODS[method](inst, time_limit, rng, **params)
    elapsed = time.perf_counter() - start
    valid = True
    error = None
    try:
        verify(inst, res)
    except ValueError as e:
        valid = False
        error = str(e)
    return {
        "method": method,
        "instance": inst.name,
        "n": inst.n,
        "m": inst.m,
        "seed": seed,
        "time_limit": time_limit,
        "params": params,
        "size": res.size,
        "status": res.status,
        "timed_out": res.timed_out,
        "lower_bound": res.lower_bound,
        "nodes": res.nodes,
        "time": elapsed,
        "valid": valid,
        "error": error,
        "selected": res.selected,
    }
