"""Lower bounds on the size of a minimum cover.

- counting_bound: u uncovered elements and at most g of them per subset need
  at least ceil(u / g) subsets.
- lp_relaxation: optimal value z_LP of the linear relaxation, solved by HiGHS.
- regular_lp_value: z_LP without a solver when the instance is regular.
- lp_bound: ceil(z_LP), from regular_lp_value when it applies, else from HiGHS.
"""

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
    HiGHS reads its clock only once the simplex has started: building the matrix,
    the presolve and the setup of the simplex are not interrupted, so the call can
    last a few seconds longer than time_limit on large instances (STS2187).
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


def regular_lp_value(inst: Instance) -> float | None:
    """z_LP = n / p if every subset has p elements and every element is in t subsets.

    Taking 1/t of every subset covers each element exactly once, so z_LP <= m / t.
    Giving 1/p to every element puts at most 1 in each subset, a feasible solution
    of the dual, so z_LP >= n / p. Counting the pairs (e, S_j) with e in S_j gives
    n * t = m * p, so both values are equal. This holds for the STS (t = 3) and
    CYC (t = 4) families. Returns None if the instance is not regular. O(n + m).
    """
    if inst.n == 0 or inst.m == 0:
        return None
    p = len(inst.sets[0])
    t = len(inst.covering[0])
    if p == 0 or t == 0:
        return None
    if any(len(s) != p for s in inst.sets) or any(len(c) != t for c in inst.covering):
        return None
    return inst.n / p


def lp_bound(inst: Instance, time_limit: float | None = None) -> int | None:
    """ceil(z_LP), a lower bound on the optimum, or None if time_limit is reached first.

    On a regular instance z_LP is known exactly and HiGHS is not called.
    """
    value = regular_lp_value(inst)
    if value is None:
        value = lp_relaxation(inst, time_limit)
    if value is None:
        return None
    return math.ceil(value - 1e-6)


def counting_bound(inst: Instance, uncovered: set[int] | None = None) -> int:
    if uncovered is None:
        # Every element is uncovered: the best gain is the largest subset, O(m).
        if inst.n == 0:
            return 0
        return math.ceil(inst.n / max(len(s) for s in inst.sets))
    if not uncovered:
        return 0
    best_gain = max(sum(1 for e in s if e in uncovered) for s in inst.sets)
    return math.ceil(len(uncovered) / best_gain)