"""Brute force for the unweighted minimum set cover.

Try every combination of k subsets for k = 0, 1, 2, ... and stop at the first
combination that covers all the elements. All the smaller sizes have been
tried before, so this cover is optimal.

The union of the chosen subsets is built incrementally: going one level deeper
adds one subset to the union of the previous ones, so each combination costs a
single OR instead of k.

Number of combinations examined: at most sum_{k=0}^{OPT} C(m, k).
If the time limit stops the search while trying size k, every size below k has
been tried without success, so OPT >= k: this is returned as lower_bound.
"""

import time

import numpy as np

from setcover.instance import Instance
from setcover.result import Result

CHECK_EVERY = 1 << 14  # check the deadline every 16384 combinations


def solve(inst: Instance, time_limit: float, rng: np.random.Generator) -> Result:
    deadline = time.perf_counter() + time_limit
    masks = inst.masks
    full = (1 << inst.n) - 1
    k = 0
    cover = None
    timed_out = False
    checked = 0
    while cover is None and not timed_out and k <= inst.m:
        cover, n_checked, timed_out = _search_size(masks, full, k, deadline)
        checked += n_checked
        if cover is None and not timed_out:
            k += 1
    extra = {"checked": checked}
    if cover is not None:
        return Result(selected=cover, size=k, status="optimal", lower_bound=k, extra=extra)
    if timed_out:
        return Result(selected=[], size=0, status="none", timed_out=True, lower_bound=k, extra=extra)
    extra["reason"] = "infeasible"
    return Result(selected=[], size=0, status="none", extra=extra)


def _search_size(masks: tuple[int, ...], full: int, k: int, deadline: float) -> tuple[list[int] | None, int, bool]:
    """Look for k subsets whose union is full.

    Returns the cover found (or None), the number of combinations examined,
    and True if the deadline stopped the search.
    """
    m = len(masks)
    chosen = []
    checked = 0
    timed_out = False

    def extend(start: int, union: int) -> bool:
        nonlocal checked, timed_out
        if len(chosen) == k:
            checked += 1
            if checked % CHECK_EVERY == 0 and time.perf_counter() >= deadline:
                timed_out = True
            return union == full
        last_start = m - (k - len(chosen))
        j = start
        found = False
        while j <= last_start and not found and not timed_out:
            chosen.append(j)
            found = extend(j + 1, union | masks[j])
            if not found:
                chosen.pop()
            j += 1
        return found

    found = extend(0, 0)
    return (list(chosen) if found else None), checked, timed_out