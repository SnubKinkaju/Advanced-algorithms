import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from setcover.instance import Instance


def coverage_matrix(inst: Instance) -> csr_matrix:
    rows = [e for s in inst.sets for e in s]
    cols = [j for j, s in enumerate(inst.sets) for _ in s]
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(inst.n, inst.m))


def lp_relaxation(inst: Instance, time_limit: float | None = None) -> float | None:
    """Optimal value of the linear relaxation, or None if time_limit is reached first.

    A relaxation stopped before the end gives no valid bound, so nothing is returned.
    """
    options = {} if time_limit is None else {"time_limit": max(time_limit, 0.0)}
    res = linprog(
        c=np.ones(inst.m),
        A_ub=-coverage_matrix(inst),
        b_ub=-np.ones(inst.n),
        bounds=(0, 1),
        method="highs",
        options=options,
    )
    if res.status == 1:
        return None
    if res.status != 0:
        raise RuntimeError(f"{inst.name}: linear relaxation not solved ({res.message})")
    return float(res.fun)


def lp_bound(inst: Instance, time_limit: float | None = None) -> int | None:
    """ceil(z_LP), a lower bound on the optimum, or None if time_limit is reached first."""
    value = lp_relaxation(inst, time_limit)
    if value is None:
        return None
    return math.ceil(value - 1e-6)


def counting_bound(inst: Instance, uncovered: set[int] | None = None) -> int:
    if uncovered is None:
        uncovered = set(range(inst.n))
    if not uncovered:
        return 0
    best_gain = max(sum(1 for e in s if e in uncovered) for s in inst.sets)
    return math.ceil(len(uncovered) / best_gain)