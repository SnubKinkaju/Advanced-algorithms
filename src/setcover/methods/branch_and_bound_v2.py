"""Branch and bound for the unweighted minimum set cover (version 2).

Same search tree as version 1: the first upper bound is the greedy cover, and
each node branches on the uncovered element with the fewest allowed subsets.
Child i takes the i-th of these subsets and forbids the previous ones, so that
every cover is found in exactly one branch. Version 2 differs in three ways.

1. One shared state, updated and restored (no copies of large integers).
   Version 1 stores in every node two integers of n and m bits (uncovered
   elements, forbidden subsets) and shifts them for every candidate. Here a
   single state is kept for the whole search:
   - cover_count[e]: number of chosen subsets that contain element e;
   - uncovered: the list of uncovered elements, with position[e] its index,
     so that an element is removed or added back in O(1);
   - forbidden: a bytearray of m flags, so the test "is j forbidden" is O(1).
   Going down to a child adds a subset in O(|S_j|); coming back removes it in
   O(|S_j|) and gives back the subsets that the node forbade.

   The gain of every subset (number of uncovered elements it would cover) is
   also kept up to date in the same way, instead of being recomputed.

2. Two node bounds, the stronger one is used. A node at depth d is cut when
   d + bound reaches the size of the best cover.
   - Largest gains: k subsets cover at most the sum of the k largest gains
     of the allowed subsets. If this sum is still smaller than the number u
     of uncovered elements, k more subsets are not enough. This is never
     weaker than the counting bound ceil(u / g) of version 1, where every
     subset counts as the best one.
   - Disjoint elements: if r uncovered elements are pairwise never covered by
     the same allowed subset, at least r more subsets are needed, one per
     element. The elements are picked greedily: take an unmarked uncovered
     element, count it, and mark every element of the allowed subsets that
     contain it. It helps when many subsets have the same size (STS
     instances), where the largest gains are all equal.
   The first bound is cheap and strong on dense instances (E family), where
   almost all elements share a subset and the second one stays close to 1.

3. The LP bound within a share of the budget, used during the whole search.
   The LP relaxation gets at most lp_fraction of the time limit, so that most
   of the budget is left for the search (in version 1 it could take all the
   time on large instances, and the method then returned the greedy cover).
   If it is not solved in time, the root lower bound is 1. The search stops as
   soon as the best cover reaches the LP bound, since it is then optimal.

Complexity. A node costs O(N + m log m) in the worst case, where
N = sum of |S_j|: scanning the uncovered elements and their allowed subsets to
choose the branching element and for the disjoint bound, plus selecting the
largest gains among the m subsets. Adding or removing subset j costs
O(sum of |covering[e]| for e in S_j), to update the gains. The number of nodes is exponential in the worst case, as for any
exact method on an NP-hard problem. Memory is O(n + m) for the state, plus the
candidate lists of the nodes on the current path.

If the whole tree is explored, or if the best cover reaches the LP bound, the
best cover is optimal. If the time limit is reached, the best cover found so
far is returned with status "feasible".
"""

import heapq
import time

import numpy as np

from setcover.bounds import lp_bound
from setcover.instance import Instance
from setcover.methods import greedy
from setcover.result import Result


class _Timeout(Exception):
    """Raised inside the search when the deadline is reached."""


class _Search:
    """Depth-first search on one shared state, updated going down and restored going up."""

    def __init__(self, inst: Instance, deadline: float, best: list[int], lower_bound: int):
        self.inst = inst
        self.deadline = deadline
        self.best = best                     # best cover found so far
        self.lower_bound = lower_bound       # root lower bound (LP, or 1)
        self.proved = False                  # True when best reaches lower_bound

        self.cover_count = [0] * inst.n
        self.uncovered = list(range(inst.n))
        self.position = list(range(inst.n))  # position[e] = index of e in uncovered
        self.forbidden = bytearray(inst.m)
        self.gain = [len(s) for s in inst.sets]  # uncovered elements of each subset
        self.chosen = []                     # subsets on the current path

        # mark[e] == stamp means e is marked in the current disjoint bound.
        # A new stamp at each call avoids clearing the array.
        self.mark = [0] * inst.n
        self.stamp = 0

        self.nodes = 0
        self.ticks = 0

    def _tick(self):
        """Check the clock every 1024 calls, inside the loops of one node."""
        self.ticks += 1
        if self.ticks & 1023 == 0 and time.perf_counter() >= self.deadline:
            raise _Timeout

    def _add(self, j: int):
        """Take subset j: its elements become covered."""
        self.chosen.append(j)
        for e in self.inst.sets[j]:
            self.cover_count[e] += 1
            if self.cover_count[e] == 1:
                # Remove e from uncovered: move the last element to its place.
                i = self.position[e]
                last = self.uncovered[-1]
                self.uncovered[i] = last
                self.position[last] = i
                self.uncovered.pop()
                # e no longer counts in the gain of the subsets that contain it.
                for k in self.inst.covering[e]:
                    self.gain[k] -= 1

    def _remove(self, j: int):
        """Undo _add(j)."""
        self.chosen.pop()
        for e in self.inst.sets[j]:
            self.cover_count[e] -= 1
            if self.cover_count[e] == 0:
                self.position[e] = len(self.uncovered)
                self.uncovered.append(e)
                for k in self.inst.covering[e]:
                    self.gain[k] += 1

    def _branching_subsets(self) -> list[int]:
        """Allowed subsets of the uncovered element that has the fewest of them.

        Returns [] if an uncovered element has no allowed subset left.
        """
        covering, forbidden = self.inst.covering, self.forbidden
        best = None
        for e in self.uncovered:
            self._tick()
            cands = [j for j in covering[e] if not forbidden[j]]
            if not cands:
                return []
            if best is None or len(cands) < len(best):
                best = cands
                if len(cands) == 1:
                    break
        return best

    def _largest_gains_cut(self, limit: int) -> bool:
        """True if limit - 1 more subsets cannot cover the uncovered elements.

        The limit - 1 allowed subsets with the largest gains cover at most the
        sum of these gains; if it is below u, at least limit subsets are needed.
        """
        forbidden = self.forbidden
        gains = [g for j, g in enumerate(self.gain) if not forbidden[j]]
        return sum(heapq.nlargest(limit - 1, gains)) < len(self.uncovered)

    def _disjoint_bound(self, limit: int) -> int:
        """Number r of uncovered elements that pairwise share no allowed subset.

        Counting stops at limit: the node is cut anyway once r reaches it.
        """
        sets, covering, forbidden, mark = self.inst.sets, self.inst.covering, self.forbidden, self.mark
        self.stamp += 1
        stamp = self.stamp
        r = 0
        for e in self.uncovered:
            self._tick()
            if mark[e] == stamp:
                continue
            r += 1
            if r >= limit:
                return r
            # Every element that shares an allowed subset with e is marked.
            for j in covering[e]:
                if not forbidden[j]:
                    for x in sets[j]:
                        mark[x] = stamp
        return r

    def _expand(self):
        """Evaluate the current node.

        Returns the subsets to branch on, or None if the node is closed
        (complete cover, or cut by a bound).
        """
        self.nodes += 1
        if time.perf_counter() >= self.deadline:
            raise _Timeout
        depth = len(self.chosen)

        # Leaf: every element is covered.
        if not self.uncovered:
            if depth < len(self.best):
                self.best = list(self.chosen)
                if len(self.best) <= self.lower_bound:
                    self.proved = True
            return None

        # At least one more subset is needed.
        if depth + 1 >= len(self.best):
            return None

        cands = self._branching_subsets()
        if not cands:
            return None

        # Cut if at least limit more subsets are needed, i.e. if
        # depth + bound >= size of the best cover. The cheap bound first.
        limit = len(self.best) - depth
        if self._largest_gains_cut(limit):
            return None
        if self._disjoint_bound(limit) >= limit:
            return None

        # Try first the subsets that cover the most uncovered elements.
        cands.sort(key=self.gain.__getitem__, reverse=True)
        return cands

    def run(self):
        """Explore the whole tree, unless the deadline or the lower bound stops it."""
        cands = self._expand()
        if cands is None or self.proved:
            return
        # A frame is [subsets to branch on, index of the next child].
        stack = [[cands, 0]]
        while stack:
            if self.proved:
                return
            frame = stack[-1]
            cands, i = frame
            if i == len(cands) or len(self.chosen) + 1 >= len(self.best):
                # Node finished: give back the subsets it forbade (all the
                # children tried before the last one), then leave the node.
                for j in cands[:max(i - 1, 0)]:
                    self.forbidden[j] = 0
                stack.pop()
                if stack:
                    parent_cands, parent_i = stack[-1]
                    self._remove(parent_cands[parent_i - 1])
                continue
            # Child i forbids the subset of child i - 1, already explored.
            if i > 0:
                self.forbidden[cands[i - 1]] = 1
            frame[1] = i + 1
            self._add(cands[i])
            child = self._expand()
            if child is None:
                self._remove(cands[i])
            else:
                stack.append([child, 0])


def solve(inst: Instance, time_limit: float, rng: np.random.Generator,
          use_lp: bool = True, lp_fraction: float = 0.2) -> Result:
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
    best = list(start.selected)

    # Root lower bound: the LP bound, with at most lp_fraction of the time limit.
    root_lb = 1
    lp = None
    if use_lp:
        lp_time = min(lp_fraction * time_limit, deadline - time.perf_counter())
        if lp_time > 0:
            lp = lp_bound(inst, time_limit=lp_time)
            if lp is not None:
                root_lb = lp
    extra = {"lp_bound": lp}
    if len(best) <= root_lb:
        return Result(selected=sorted(best), size=len(best), status="optimal",
                      lower_bound=len(best), nodes=0, extra=extra)

    search = _Search(inst, deadline, best, root_lb)
    try:
        search.run()
        timed_out = False
    except _Timeout:
        timed_out = True

    best = search.best
    if timed_out and not search.proved:
        return Result(selected=sorted(best), size=len(best), status="feasible",
                      timed_out=True, lower_bound=root_lb, nodes=search.nodes, extra=extra)
    return Result(selected=sorted(best), size=len(best), status="optimal",
                  lower_bound=len(best), nodes=search.nodes, extra=extra)

