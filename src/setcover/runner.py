import time
from pathlib import Path

import numpy as np

from setcover.io import read_instance
from setcover.registry import METHODS
from setcover.result import verify


def run_one(method: str, instance_path: str | Path, seed: int, time_limit: float, **params) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, available: {sorted(METHODS)}")
    setup_start = time.perf_counter()
    inst = read_instance(instance_path)
    # The masks are built on first use. Build them here, before the clock
    # starts, so that methods using masks are not charged for preparing the
    # data (about 0.5 s on sts/data.2187), like the reading of the file.
    _ = inst.masks
    setup_time = time.perf_counter() - setup_start
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
        "extra": res.extra,
        "time": elapsed,
        "setup_time": setup_time,
        "valid": valid,
        "error": error,
        "selected": res.selected,
    }