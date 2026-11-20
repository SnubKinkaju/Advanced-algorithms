"""Greedy algorithm for the unweighted minimum set cover (Chvatal, 1979).

At each step, select the subset that covers the largest number of elements
that are still uncovered, until every element is covered.

The cover is valid but not always optimal: its size is at most H(d) times the
optimum, where d is the size of the largest subset and H(d) = 1 + 1/2 + ... + 1/d.

Two optional variants are available through keyword arguments:
- random_ties: when several subsets have the same largest gain, pick one of
  them at random instead of always the one with the smallest index.
- remove_redundant: at the end, remove the selected subsets that are not
  needed any more because the other selected subsets cover all their elements.

Complexity: O(k * m + N) time and O(n + m) memory, where k is the size of the
cover and N the sum of the sizes of all the subsets. Each of the k rounds
scans the m gains to find the largest one. The gain updates cost O(N) in
total, because each element is covered for the first time only once.
remove_redundant adds O(N).
"""

import time

import numpy as np

from setcover.instance import Instance
from setcover.result import Result


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator,
    random_ties: bool = False,
    remove_redundant: bool = False,
) -> Result:
    deadline = time.perf_counter() + time_limit

    # gain[j] = number of still uncovered elements that subset j would cover.
    # At the start nothing is covered, so the gain is the size of the subset.
    gain = [len(s) for s in inst.sets]
    covered = [False] * inst.n
    n_uncovered = inst.n
    selected = []

    # Largest gain. Gains only decrease, so this value never goes up: it is
    # kept from one round to the next and recomputed only when needed.
    best_gain = max(gain, default=0)

    while n_uncovered > 0:
        if time.perf_counter() >= deadline:
            # The cover is not complete yet, so there is nothing valid to return.
            return Result(selected=[], size=0, status="none", timed_out=True)

        if best_gain not in gain:
            # No subset has this gain any more: find the new largest gain.
            best_gain = max(gain, default=0)
        if best_gain == 0:
            # No subset covers any remaining element: the instance has no cover.
            return Result(selected=[], size=0, status="none", extra={"reason": "infeasible"})

        # Greedy choice: a subset with the largest gain.
        if random_ties:
            # Variant: choose at random among all the subsets with this gain.
            # Draw a rank among the ties, then move to the tie of that rank.
            # No list of the ties is built: count() and index() scan the
            # gains much faster than a Python loop.
            rank = int(rng.integers(gain.count(best_gain)))
            best = gain.index(best_gain)
            for _ in range(rank):
                best = gain.index(best_gain, best + 1)
        else:
            # Default: the smallest index, so the result is always the same.
            best = gain.index(best_gain)

        selected.append(best)

        # Mark the new elements as covered and update the gains.
        # Each newly covered element e no longer counts for any subset that
        # contains it, so the gain of these subsets decreases by one.
        for e in inst.sets[best]:
            if not covered[e]:
                covered[e] = True
                n_uncovered -= 1
                for j in inst.covering[e]:
                    gain[j] -= 1

    greedy_size = len(selected)
    timed_out = False
    if remove_redundant:
        selected, timed_out = _remove_redundant(inst, selected, deadline)

    # The cover is valid, but the greedy gives no proof of optimality.
    return Result(
        selected=selected,
        size=len(selected),
        status="feasible",
        timed_out=timed_out,
        extra={"removed": greedy_size - len(selected)},
    )


def _remove_redundant(inst: Instance, selected: list[int], deadline: float) -> tuple[list[int], bool]:
    """Remove the subsets that the rest of the cover makes useless.

    A selected subset is redundant when each of its elements is also covered
    by another selected subset. Subsets are examined in the order of their
    selection: the first ones are the largest, and the subsets selected after
    them may have covered all their elements again.

    Returns the new cover and True if the deadline stopped the cleaning.
    """
    # count[e] = number of selected subsets that cover element e.
    count = [0] * inst.n
    for j in selected:
        for e in inst.sets[j]:
            count[e] += 1

    kept = []
    for position, j in enumerate(selected):
        if time.perf_counter() >= deadline:
            # Out of time: keep the subsets that were not examined.
            return kept + selected[position:], True
        if all(count[e] >= 2 for e in inst.sets[j]):
            # Every element of j stays covered without j: remove it.
            for e in inst.sets[j]:
                count[e] -= 1
        else:
            kept.append(j)
    return kept, False
