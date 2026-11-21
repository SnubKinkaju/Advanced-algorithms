"""Brute force for the unweighted minimum set cover.

Try every combination of k subsets for k = 0, 1, 2, ... and stop at the first
combination that covers all the elements. All the smaller sizes have been
tried before, so this cover is optimal. No cover is known before that point,
so an interrupted search returns no cover (the method is not anytime).

The union of the chosen subsets is built incrementally: going one level deeper
adds one subset to the union of the previous ones, so each node of the search
costs a single OR.

Number of combinations examined: at most sum_{k=0}^{OPT} C(m, k).
If the time limit stops the search while trying size k, every size below k has
been tried without success, so OPT >= k: this is returned as lower_bound.

Time limit: one node counter for the whole search (all sizes), and the clock
is read every CHECK_EVERY nodes, before each size and at the start. Each node
does an OR, so the delay after the deadline is at most CHECK_EVERY ORs.
extra records "checked" (combinations of size k examined) and "nodes" (all
nodes visited), a measure of work that does not depend on the machine.
"""

import time

import numpy as np

from setcover.instance import Instance
from setcover.result import Result

CHECK_EVERY = 1 << 10  # read the clock every 1024 nodes (a power of two)


class _Budget:
    """Counters shared by every size, and the deadline."""

    def __init__(self, deadline: float):
        self.deadline = deadline
        self.nodes = 0
        self.checked = 0
        self.timed_out = False

    def expired(self) -> bool:
        if not self.timed_out and time.perf_counter() >= self.deadline:
            self.timed_out = True
        return self.timed_out


def solve(inst: Instance, time_limit: float, rng: np.random.Generator) -> Result:
    budget = _Budget(time.perf_counter() + time_limit)
    k = 0
    if budget.expired():
        return _result(None, k, budget)

    masks = inst.masks
    full = (1 << inst.n) - 1
    cover = None
    while k <= inst.m:
        if budget.expired():
            break
        cover = _search_size(masks, full, k, budget)
        if cover is not None or budget.timed_out:
            break
        k += 1
    return _result(cover, k, budget)


def _result(cover: list[int] | None, k: int, budget: _Budget) -> Result:
    extra = {"checked": budget.checked, "nodes": budget.nodes}
    if cover is not None:
        return Result(selected=cover, size=k, status="optimal", lower_bound=k, extra=extra)
    if budget.timed_out:
        return Result(selected=[], size=0, status="none", timed_out=True, lower_bound=k,
                      extra=extra)
    extra["reason"] = "infeasible"
    return Result(selected=[], size=0, status="none", extra=extra)


def _search_size(masks: tuple[int, ...], full: int, k: int, budget: _Budget) -> list[int] | None:
    """Look for k subsets whose union is full; None if none or out of time."""
    m = len(masks)
    chosen = []
    # Local copies of the shared counters (faster than attributes), written back at the end.
    nodes, checked, timed_out = budget.nodes, budget.checked, False
    deadline, clock, step = budget.deadline, time.perf_counter, CHECK_EVERY - 1

    def extend(start: int, union: int) -> bool:
        nonlocal nodes, checked, timed_out
        nodes += 1
        if nodes & step == 0 and clock() >= deadline:
            # The callers stop as soon as they see timed_out.
            timed_out = True
            return False
        if len(chosen) == k:
            checked += 1
            return union == full
        last_start = m - (k - len(chosen))
        for j in range(start, last_start + 1):
            chosen.append(j)
            if extend(j + 1, union | masks[j]):
                return True
            chosen.pop()
            if timed_out:
                return False
        return False

    found = extend(0, 0)
    budget.nodes, budget.checked = nodes, checked
    budget.timed_out = budget.timed_out or timed_out
    return list(chosen) if found else None