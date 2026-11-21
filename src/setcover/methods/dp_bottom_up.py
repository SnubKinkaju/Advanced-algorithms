"""Bottom-up dynamic programming for the unweighted minimum set cover.

A state is a set of covered elements, stored as a bitmask of n bits.
dp[mask] is the smallest number of subsets whose union is exactly mask.
Starting from dp[0] = 0, every state is extended by every subset j:

    dp[mask | S_j] = min(dp[mask | S_j], dp[mask] + 1).

Since mask | S_j >= mask, the states are processed in increasing order and
every state is final when it is reached. The answer is dp[2^n - 1], and the
cover is rebuilt by following the parent of each state.

Complexity: Theta(m * 2^n) time and Theta(2^n) memory (three lists of 2^n
entries). The method is only usable for small n: instances with
2^n > max_states are refused at once with reason "too_large".

The clock is checked once per state, not once per transition: a transition is
only an OR and a comparison, and reading the clock costs about as much.
After the deadline, at most m more transitions are done. If the full state has
already been reached, the cover found so far is returned as "feasible".
"""

import time

import numpy as np

from setcover.instance import Instance
from setcover.result import Result


def _rebuild(parent: list[int], chosen: list[int], full_mask: int) -> list[int]:
    """Follow the parents from the full state back to the empty one.

    Valid at any time of the search: parent[x] < x, and each step adds
    exactly the subset chosen[x], so the union of the subsets is full_mask.
    """
    selected = []
    mask = full_mask
    while mask != 0:
        selected.append(chosen[mask])
        mask = parent[mask]
    selected.reverse()
    return selected


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator,
    max_states: int = 1 << 20,
) -> Result:
    deadline = time.perf_counter() + time_limit

    # An element covered by no subset: no cover exists.
    if any(not c for c in inst.covering):
        return Result(selected=[], size=0, status="none", extra={"reason": "infeasible"})

    n = inst.n
    total_states = 1 << n
    full_mask = total_states - 1

    # The table needs 2^n entries: refuse the instance before allocating it.
    if total_states > max_states:
        return Result(selected=[], size=0, status="none", extra={"reason": "too_large"})

    if time.perf_counter() >= deadline:
        return Result(selected=[], size=0, status="none", timed_out=True)

    dp = [float("inf")] * total_states
    parent = [-1] * total_states
    chosen = [-1] * total_states
    dp[0] = 0
    masks = inst.masks

    for mask in range(total_states):
        if time.perf_counter() >= deadline:
            # Keep the cover already found for the full state, if any.
            if dp[full_mask] == float("inf"):
                return Result(selected=[], size=0, status="none", timed_out=True)
            selected = _rebuild(parent, chosen, full_mask)
            return Result(selected=selected, size=len(selected), status="feasible",
                          timed_out=True)

        if dp[mask] == float("inf"):
            # This set of elements cannot be reached exactly.
            continue

        new_cost = dp[mask] + 1
        for j, subset_mask in enumerate(masks):
            new_mask = mask | subset_mask
            if new_cost < dp[new_mask]:
                dp[new_mask] = new_cost
                parent[new_mask] = mask
                chosen[new_mask] = j

    # Safety net: with every element covered by a subset, this cannot happen.
    if dp[full_mask] == float("inf"):
        return Result(selected=[], size=0, status="none", extra={"reason": "infeasible"})

    selected = _rebuild(parent, chosen, full_mask)
    return Result(selected=selected, size=len(selected), status="optimal",
                  lower_bound=len(selected))