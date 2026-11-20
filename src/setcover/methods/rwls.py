"""RWLS (Gao, Yao, Weise, Li, EJOR 2015) for the unweighted minimum set cover.

Elements have weights (start at 1). Score of a subset outside the solution:
weight of the uncovered elements it covers; inside: minus the weight of the
elements only it covers. Start from the random-tie greedy. When the solution
is a cover, record it and remove one subset. Otherwise remove the best
non-tabu subset, add the best allowed subset covering a random uncovered
element, then add 1 to the weight of every uncovered element.

Complexity: O(t * p) per add or remove (t = max subsets per element,
p = max subset size), plus O(|C|) to choose the removal. Memory O(n + m).
"""

import math
import time
from collections import deque
from collections.abc import Iterable

import numpy as np

from setcover.instance import Instance
from setcover.methods import greedy
from setcover.result import Result


class _IndexedSet:
    """Integers in [0, size): O(1) add, remove, membership and random draw."""

    def __init__(self, size: int, items: Iterable[int] = ()):
        self.items: list[int] = []
        self.pos = [-1] * size
        for x in items:
            self.add(x)

    def add(self, x: int) -> None:
        self.pos[x] = len(self.items)
        self.items.append(x)

    def remove(self, x: int) -> None:
        i = self.pos[x]
        last = self.items.pop()
        if last != x:
            self.items[i] = last
            self.pos[last] = i
        self.pos[x] = -1

    def __contains__(self, x: int) -> bool:
        return self.pos[x] >= 0

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __sub__(self, other) -> set[int]:
        return {x for x in self.items if x not in other}


class _SearchState:
    """Search data, separate from the shared instance."""

    def __init__(self, inst: Instance):
        self.selected = _IndexedSet(inst.m)
        self.cover_count = [0] * inst.n
        self.uncovered = _IndexedSet(inst.n, range(inst.n))
        self.weights = [1] * inst.n
        self.scores = [len(subset) for subset in inst.sets]
        self.can_add = [True] * inst.m
        self.timestamps = [0] * inst.m
        # Only subset covering some element: never removed.
        self.fixed = {columns[0] for columns in inst.covering if len(columns) == 1}
        self.is_fixed = [False] * inst.m
        for j in self.fixed:
            self.is_fixed[j] = True


def _choose_subset(
    candidates: Iterable[int],
    state: _SearchState,
    rng: np.random.Generator,
    use_age: bool = True,
) -> int:
    """Highest score, then oldest move, then random."""
    best_key = None
    ties = []
    for j in candidates:
        age = -state.timestamps[j] if use_age else 0
        key = (state.scores[j], age)
        if best_key is None or key > best_key:
            best_key = key
            ties = [j]
        elif key == best_key:
            ties.append(j)
    return ties[int(rng.integers(len(ties)))]


def _add_subset(inst: Instance, state: _SearchState, j: int) -> None:
    state.selected.add(j)
    state.scores[j] = -state.scores[j]
    for e in inst.sets[j]:
        state.cover_count[e] += 1
        count = state.cover_count[e]
        if count == 1:
            state.uncovered.remove(e)
        for other in inst.covering[e]:
            if other == j:
                continue
            state.can_add[other] = True
            if count == 1:
                state.scores[other] -= state.weights[e]
            elif count == 2 and other in state.selected:
                state.scores[other] += state.weights[e]


def _remove_subset(inst: Instance, state: _SearchState, j: int) -> None:
    state.selected.remove(j)
    state.scores[j] = -state.scores[j]
    for e in inst.sets[j]:
        state.cover_count[e] -= 1
        count = state.cover_count[e]
        if count == 0:
            state.uncovered.add(e)
        for other in inst.covering[e]:
            if other == j:
                continue
            state.can_add[other] = True
            if count == 0:
                state.scores[other] += state.weights[e]
            elif count == 1 and other in state.selected:
                state.scores[other] -= state.weights[e]
    state.can_add[j] = False


def _increase_weights(inst: Instance, state: _SearchState) -> None:
    for e in state.uncovered:
        state.weights[e] += 1
        for j in inst.covering[e]:
            state.scores[j] += 1


def _removable(state: _SearchState) -> list[int]:
    is_fixed = state.is_fixed
    return [j for j in state.selected if not is_fixed[j]]


def _initial_cover(
    inst: Instance,
    state: _SearchState,
    deadline: float,
    rng: np.random.Generator,
) -> bool:
    """Random-tie greedy, the initialization of the article. False on timeout."""
    start = greedy.solve(inst, max(0.0, deadline - time.perf_counter()), rng,
                         random_ties=True)
    if start.status == "none":
        return False
    _load_cover(inst, state, start.selected)
    return True


def _load_cover(inst: Instance, state: _SearchState, cover: list[int]) -> None:
    """Same state as calling _add_subset on each subset, but in O(sum |S_j|)."""
    for j in cover:
        state.selected.add(j)
        for e in inst.sets[j]:
            state.cover_count[e] += 1
    state.uncovered = _IndexedSet(inst.n, (e for e in range(inst.n) if state.cover_count[e] == 0))
    count, weights = state.cover_count, state.weights
    for j, subset in enumerate(inst.sets):
        if j in state.selected:
            state.scores[j] = -sum(weights[e] for e in subset if count[e] == 1)
        else:
            state.scores[j] = sum(weights[e] for e in subset if count[e] == 0)


def _exchange_subset(
    inst: Instance,
    state: _SearchState,
    tabu: deque[int],
    step: int,
    rng: np.random.Generator,
) -> bool:
    removable = _removable(state)
    if not removable:
        return False

    # The two subsets added last are tabu.
    candidates = [j for j in removable if j not in tabu]
    while not candidates:
        tabu.popleft()
        candidates = [j for j in removable if j not in tabu]
    removed = _choose_subset(candidates, state, rng)
    _remove_subset(inst, state, removed)

    uncovered = state.uncovered.items
    e = uncovered[int(rng.integers(len(uncovered)))]
    candidates = [j for j in inst.covering[e] if state.can_add[j]]
    added = _choose_subset(candidates, state, rng)
    _add_subset(inst, state, added)
    _increase_weights(inst, state)

    tabu.append(added)
    state.timestamps[removed] = step
    state.timestamps[added] = step
    return True


def _local_search(
    inst: Instance,
    state: _SearchState,
    begin: float,
    deadline: float,
    rng: np.random.Generator,
    max_steps: int | None,
) -> Result:
    best_selected = sorted(state.selected)
    best_time = time.perf_counter() - begin
    best_step = 0
    tabu: deque[int] = deque(maxlen=2)
    steps = 0

    while time.perf_counter() < deadline:
        if not state.uncovered:
            if len(state.selected) < len(best_selected):
                best_selected = sorted(state.selected)
                best_time = time.perf_counter() - begin
                best_step = steps
            removable = _removable(state)
            if not removable or len(best_selected) == 1:
                break
            j = _choose_subset(removable, state, rng, use_age=False)
            _remove_subset(inst, state, j)
            continue

        if max_steps is not None and steps >= max_steps:
            break
        if not _exchange_subset(inst, state, tabu, steps + 1, rng):
            break
        steps += 1

    if not state.uncovered and len(state.selected) < len(best_selected):
        best_selected = sorted(state.selected)
        best_time = time.perf_counter() - begin
        best_step = steps

    return Result(
        selected=best_selected,
        size=len(best_selected),
        status="feasible",
        timed_out=time.perf_counter() >= deadline,
        extra={"steps": steps, "best_step": best_step, "best_time": best_time},
    )


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator,
    max_steps: int | None = None,
) -> Result:
    begin = time.perf_counter()
    deadline = begin + time_limit

    if not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be finite and non-negative")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 0):
        raise ValueError("max_steps must be a non-negative integer or None")
    if inst.n == 0:
        return Result(selected=[], size=0, status="optimal", lower_bound=0)
    if time.perf_counter() >= deadline:
        return Result(selected=[], size=0, status="none", timed_out=True)
    if any(not columns for columns in inst.covering):
        return Result(selected=[], size=0, status="none", extra={"reason": "infeasible"})

    state = _SearchState(inst)
    if not _initial_cover(inst, state, deadline, rng):
        return Result(selected=[], size=0, status="none", timed_out=True)

    return _local_search(inst, state, begin, deadline, rng, max_steps)