import time

import numpy as np

from setcover.instance import Instance
from setcover.result import Result


def solve(inst: Instance, time_limit: float, rng: np.random.Generator) -> Result:
    deadline = time.perf_counter() + time_limit
    covered = [False] * inst.n
    n_covered = 0
    selected = []
    j = 0
    while n_covered < inst.n and j < inst.m and time.perf_counter() < deadline:
        new = [e for e in inst.sets[j] if not covered[e]]
        if new:
            selected.append(j)
            for e in new:
                covered[e] = True
            n_covered += len(new)
        j += 1
    if n_covered < inst.n:
        return Result(selected=[], size=0, status="none", timed_out=True)
    return Result(selected=selected, size=len(selected), status="feasible")
