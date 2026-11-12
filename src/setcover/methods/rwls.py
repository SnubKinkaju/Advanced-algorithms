
import math
import time
from collections import deque
from collections.abc import Iterable

import numpy as np

from setcover.instance import Instance
from setcover.result import Result


class _SearchState:
    """Mutable search data, kept separate from the shared instance."""

    def __init__(self, inst: Instance):
        self.selected: set[int] = set()
        self.cover_count = [0] * inst.n
        self.uncovered = set(range(inst.n))
        self.weights = [1] * inst.n
        self.scores = [len(subset) for subset in inst.sets]
        self.can_add = [True] * inst.m
        self.timestamps = [0] * inst.m
        # These subsets are the only way to cover at least one element.
        self.fixed = {columns[0] for columns in inst.covering if len(columns) == 1}


def _choose_subset(
    candidates: Iterable[int],
    state: _SearchState,
    rng: np.random.Generator,
    use_age: bool = True,
) -> int:
    """Prefer the highest score, then the oldest move, then a random tie."""
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
    """Add a subset and update coverage, scores and neighboring tabu flags."""
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
                # This element no longer contributes to an addition gain.
                state.scores[other] -= state.weights[e]
            elif count == 2 and other in state.selected:
                # The previous sole covering subset can now be removed safely.
                state.scores[other] += state.weights[e]


def _remove_subset(inst: Instance, state: _SearchState, j: int) -> None:
    """Remove a subset and update the same data as the addition operator."""
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
    # A removed subset becomes eligible again after a neighbor changes.
    state.can_add[j] = False


def _increase_weights(inst: Instance, state: _SearchState) -> None:
    """Give uncovered elements more influence on the next choices."""
    for e in state.uncovered:
        state.weights[e] += 1
        for j in inst.covering[e]:
            state.scores[j] += 1


def _initial_cover(
    inst: Instance,
    state: _SearchState,
    deadline: float,
    rng: np.random.Generator,
) -> bool:
    """Build a greedy cover; return False if the deadline interrupts it."""
    while state.uncovered:
        if time.perf_counter() >= deadline:
            return False
        candidates = (j for j in range(inst.m) if j not in state.selected)
        j = _choose_subset(candidates, state, rng, use_age=False)
        if time.perf_counter() >= deadline:
            return False
        _add_subset(inst, state, j)
    return True


def _exchange_subset(
    inst: Instance,
    state: _SearchState,
    tabu: deque[int],
    step: int,
    rng: np.random.Generator,
) -> bool:
    """Exchange one subset while keeping the candidate size unchanged."""
    removable = sorted(state.selected - state.fixed)
    if not removable:
        return False

    # Normally, the two most recently added subsets cannot be removed.
    candidates = [j for j in removable if j not in tabu]
    while not candidates:
        tabu.popleft()
        candidates = [j for j in removable if j not in tabu]
    removed = _choose_subset(candidates, state, rng)
    _remove_subset(inst, state, removed)

    # Repair a randomly chosen uncovered element using an eligible subset.
    uncovered = sorted(state.uncovered)
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
    deadline: float,
    rng: np.random.Generator,
    max_steps: int | None,
) -> Result:
    """Search for smaller covers and always preserve the best valid one."""
    best_selected = sorted(state.selected)
    tabu: deque[int] = deque(maxlen=2)
    steps = 0

    while time.perf_counter() < deadline:
        # Shrink a valid cover until some elements become uncovered.
        if not state.uncovered:
            best_selected = sorted(state.selected)
            removable = sorted(state.selected - state.fixed)
            if not removable or len(best_selected) == 1:
                break
            j = _choose_subset(removable, state, rng, use_age=False)
            _remove_subset(inst, state, j)
            continue

        # An optional step limit makes debugging runs reproducible.
        if max_steps is not None and steps >= max_steps:
            break
        if not _exchange_subset(inst, state, tabu, steps + 1, rng):
            break
        steps += 1
        if not state.uncovered:
            best_selected = sorted(state.selected)

    return Result(
        selected=best_selected,
        size=len(best_selected),
        status="feasible",
        timed_out=time.perf_counter() >= deadline,
        extra={"steps": steps},
    )


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator,
    max_steps: int | None = None,
) -> Result:
    """Run RWLS; max_steps optionally limits exchanges for debugging."""
    deadline = time.perf_counter() + time_limit

    # 1. Check the budget and handle special cases.
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

    # 2. Build the first valid cover without changing the instance.
    state = _SearchState(inst)
    if not _initial_cover(inst, state, deadline, rng):
        return Result(selected=[], size=0, status="none", timed_out=True)

    # 3. Improve it and return the best valid cover found.
    return _local_search(inst, state, deadline, rng, max_steps)
