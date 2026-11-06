import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from setcover.bounds import coverage_matrix
from setcover.instance import Instance
from setcover.result import Result


def solve(inst: Instance, time_limit: float, rng: np.random.Generator) -> Result:
    res = milp(
        c=np.ones(inst.m),
        constraints=LinearConstraint(coverage_matrix(inst), lb=1, ub=np.inf),
        integrality=np.ones(inst.m),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "disp": False},
    )
    dual_bound = getattr(res, "mip_dual_bound", None)
    lower_bound = None if dual_bound is None else math.ceil(dual_bound - 1e-6)
    if res.x is None:
        return Result(selected=[], size=0, status="none", timed_out=res.status == 1, lower_bound=lower_bound)
    selected = [j for j in range(inst.m) if res.x[j] > 0.5]
    if res.status == 0:
        return Result(selected=selected, size=len(selected), status="optimal", lower_bound=len(selected))
    return Result(selected=selected, size=len(selected), status="feasible", timed_out=True, lower_bound=lower_bound)
