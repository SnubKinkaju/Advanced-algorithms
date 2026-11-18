"""Branch and bound for the unweighted minimum set cover (version 1).

The search starts from the greedy cover, which gives a first upper bound.
Each node of the tree is a partial solution. To branch, we pick the uncovered
element with the fewest allowed subsets: any cover must contain one of them,
so child i takes the i-th subset and forbids the previous ones. This way every
cover is found in exactly one branch.

A node is cut when it cannot lead to a smaller cover than the best one found:
- at least one more subset is needed while elements remain uncovered;
- u uncovered elements and at most g of them per allowed subset need at least
  ceil(u / g) more subsets (counting bound);
- an element that no allowed subset covers makes the branch infeasible.

The LP bound is used once, at the root. If the whole tree is explored, the best
cover is optimal. If the time limit is reached, the best cover found is returned.
"""

import math
import time

import numpy as np

from setcover.bounds import lp_bound
from setcover.instance import Instance
from setcover.methods import greedy
from setcover.result import Result


def _bits(x: int):
    """Yield the positions of the bits set to 1 in x, i.e. the elements of a mask.

    The bits are read from the binary string of x, which costs O(n) in total.
    Clearing the lowest bit at each step would rebuild x every time,
    which costs O(n) per bit.
    """
    bits = bin(x)[:1:-1]
    i = bits.find("1")
    while i != -1:
        yield i
        i = bits.find("1", i + 1)


def _unwind(chosen) -> list[int]:
    """Turn the linked list of chosen subsets into a Python list."""
    out = []
    while chosen is not None:
        j, chosen = chosen
        out.append(j)
    return out


def solve(inst: Instance, time_limit: float, rng: np.random.Generator,
          use_lp: bool = True) -> Result:
    deadline = time.perf_counter() + time_limit

    # An element covered by no subset makes the instance infeasible.
    if any(not c for c in inst.covering):
        return Result(selected=[], size=0, status="none", extra={"reason": "infeasible"})
    if inst.n == 0:
        return Result(selected=[], size=0, status="optimal", lower_bound=0, nodes=0)

    # Initial upper bound: the shared greedy algorithm, within the remaining time.
    start = greedy.solve(inst, deadline - time.perf_counter(), rng)
    if start.status == "none":
        return Result(selected=[], size=0, status="none", timed_out=True, nodes=0)
    best = start.selected

    # Root lower bound: the LP bound if it is computed in time, otherwise 1.
    root_lb = 1
    if use_lp:
        lp = lp_bound(inst, time_limit=deadline - time.perf_counter())
        if lp is not None:
            root_lb = lp
    if len(best) <= root_lb:
        return Result(selected=sorted(best), size=len(best), status="optimal",
                      lower_bound=len(best), nodes=0)

    # No time left to explore: return the greedy cover without building the masks.
    if time.perf_counter() >= deadline:
        return Result(selected=sorted(best), size=len(best), status="feasible",
                      timed_out=True, lower_bound=root_lb, nodes=0)

    masks = inst.masks
    full = (1 << inst.n) - 1

    # Depth-first search with an explicit stack.
    # A node is (uncovered elements, chosen subsets, depth, forbidden subsets).
    stack = [(full, None, 0, 0)]
    nodes = 0
    timed_out = False

    while stack:
        if time.perf_counter() >= deadline:
            timed_out = True
            break

        uncovered, chosen, depth, forbidden = stack.pop()
        nodes += 1

        # Leaf: every element is covered.
        if uncovered == 0:
            if depth < len(best):
                best = _unwind(chosen)
            continue

        # At least one more subset is needed: stop if that cannot beat the best cover.
        if depth + 1 >= len(best):
            continue

        # Branch on the uncovered element with the fewest allowed subsets.
        branch = None
        for e in _bits(uncovered):
            cands = [j for j in inst.covering[e] if not (forbidden >> j) & 1]
            if not cands:
                # No allowed subset covers e: this branch has no solution.
                branch = []
                break
            if branch is None or len(cands) < len(branch):
                branch = cands
                if len(cands) == 1:
                    break
        if not branch:
            continue

        # Counting bound: u uncovered elements, at most max_gain per allowed subset.
        max_gain = max((masks[j] & uncovered).bit_count()
                       for j in range(inst.m) if not (forbidden >> j) & 1)
        if depth + math.ceil(uncovered.bit_count() / max_gain) >= len(best):
            continue

        # Try first the subsets that cover the most uncovered elements.
        branch.sort(key=lambda j: (masks[j] & uncovered).bit_count(), reverse=True)

        # Child i takes branch[i] and forbids branch[0..i-1], which were already
        # explored, so that the same cover is never visited twice.
        children = []
        f = forbidden
        for j in branch:
            children.append((uncovered & ~masks[j], (j, chosen), depth + 1, f))
            f |= 1 << j
        stack.extend(reversed(children))

    if timed_out:
        return Result(selected=sorted(best), size=len(best), status="feasible",
                      timed_out=True, lower_bound=root_lb, nodes=nodes)
    return Result(selected=sorted(best), size=len(best), status="optimal",
                  lower_bound=len(best), nodes=nodes)


