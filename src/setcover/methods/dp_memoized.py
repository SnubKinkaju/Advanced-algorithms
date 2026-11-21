"""Top-down dynamic programming (memoization) for the unweighted minimum set cover.

A state is a set of covered elements, stored as a bitmask of n bits.
g(mask) is the smallest number of subsets still needed to cover the elements
missing from mask:

    g(full) = 0,    g(mask) = 1 + min over j with S_j not inside mask of g(mask | S_j).

g is computed recursively from g(0), and each value is stored the first time
it is computed (memo), so a state is never solved twice. Only the states that
can be reached from the empty set are visited, which can be far fewer than 2^n.
The cover is rebuilt by following the subset chosen in each state.

Complexity: O(m * R) time and O(R) memory, where R <= 2^n is the number of
reachable states. Each call adds at least one element, so the recursion depth
is at most n + 1.

Limits:
- max_depth: instances with n > max_depth are refused at once with reason
  "too_large", because Python limits the depth of recursion (1000 calls by
  default). max_depth must stay below this limit.
- max_states: if the memo reaches this size, the search stops with reason
  "memory_limit".
If the search stops early (time or memory), the best complete cover seen so
far is returned as "feasible".

The clock is checked once per new state, not on every call: most calls only
read a value that is already stored.
"""

import time

import numpy as np

from setcover.instance import Instance
from setcover.result import Result


class _Timeout(Exception):
    """Raised inside the recursion when the deadline is reached."""


class _MemoryLimit(Exception):
    """Raised inside the recursion when the memo reaches max_states."""


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator,
    max_states: int = 1 << 20,
    max_depth: int = 500,
) -> Result:
    deadline = time.perf_counter() + time_limit

    # An element covered by no subset: no cover exists.
    if any(not c for c in inst.covering):
        return Result(selected=[], size=0, status="none", extra={"reason": "infeasible"})

    n = inst.n
    full_mask = (1 << n) - 1

    # The recursion can go n + 1 levels deep.
    if n > max_depth:
        return Result(selected=[], size=0, status="none", extra={"reason": "too_large"})

    if time.perf_counter() >= deadline:
        return Result(selected=[], size=0, status="none", timed_out=True)

    masks = inst.masks
    memo = {}
    choice = {}
    path = []          # subsets chosen from the empty state to the current one
    best_cover = None  # best complete cover seen so far

    def rebuild(mask):
        # Valid once g(mask) has returned: follow the stored choices.
        selected = []
        while mask != full_mask:
            j = choice[mask]
            selected.append(j)
            mask |= masks[j]
        return selected

    def g(mask):
        nonlocal best_cover
        if mask == full_mask:
            return 0
        if mask in memo:
            return memo[mask]

        # A new state: this is where the real work is done.
        if time.perf_counter() >= deadline:
            raise _Timeout
        if len(memo) >= max_states:
            raise _MemoryLimit

        best = float("inf")
        for j, subset_mask in enumerate(masks):
            new_mask = mask | subset_mask
            if new_mask == mask:
                # S_j adds no new element.
                continue
            path.append(j)
            cost = 1 + g(new_mask)
            path.pop()
            if cost < best:
                best = cost
                choice[mask] = j
                # A complete cover: the path, then j, then the stored choices.
                # (A cost that does not improve best cannot improve it either.)
                if best_cover is None or len(path) + cost < len(best_cover):
                    best_cover = path + [j] + rebuild(new_mask)

        memo[mask] = best
        return best

    try:
        optimal = g(0)
    except _Timeout:
        if best_cover is None:
            return Result(selected=[], size=0, status="none", timed_out=True)
        return Result(selected=best_cover, size=len(best_cover), status="feasible",
                      timed_out=True)
    except _MemoryLimit:
        if best_cover is None:
            return Result(selected=[], size=0, status="none",
                          extra={"reason": "memory_limit"})
        return Result(selected=best_cover, size=len(best_cover), status="feasible",
                      extra={"reason": "memory_limit"})

    # Safety net: with every element covered by a subset, this cannot happen.
    if optimal == float("inf"):
        return Result(selected=[], size=0, status="none", extra={"reason": "infeasible"})

    selected = rebuild(0)
    return Result(selected=selected, size=len(selected), status="optimal",
                  lower_bound=len(selected))