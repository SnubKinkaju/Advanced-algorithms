"""Simulated annealing on feasible covers for the unweighted minimum set cover.

Move: remove a random non-fixed subset j, repair each element left uncovered
(random element e, add the subset covering e that covers the most uncovered
elements, ties at random), then remove the subsets made redundant. The size
change delta is accepted if delta <= 0, else with probability exp(-delta / T)
(Kirkpatrick, Gelatt, Vecchi, 1983); a rejected move is undone.

Schedule (Johnson, Aragon, McGeoch, Schevon, 1989): T starts at 1 and is
multiplied by 0.95 after each level of |C| moves. When fewer than 2% of the
moves are accepted for 5 levels in a row, the search is frozen and T is reset
to its initial value, until the time limit.

Complexity: O(t * p^2) per move, t = max subsets per element, p = max subset
size (repair and cleanup both evaluate O(t * p) subsets of size p). Memory
O(n + m).
"""

import math
import time

import numpy as np

from setcover.instance import Instance
from setcover.methods.greedy import solve as greedy_cover
from setcover.result import Result


def _validate_parameters(
    time_limit: float,
    initial_temperature: float,
    cooling_rate: float,
    steps_per_temperature: int | None,
    min_acceptance: float,
    frozen_levels: int,
    max_steps: int | None,
) -> None:
    if not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be finite and non-negative")
    if not math.isfinite(initial_temperature) or initial_temperature <= 0:
        raise ValueError("initial_temperature must be finite and positive")
    if not math.isfinite(cooling_rate) or not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1 (exclusive)")
    if not math.isfinite(min_acceptance) or not 0 <= min_acceptance <= 1:
        raise ValueError("min_acceptance must be between 0 and 1")
    for name, value, minimum in (
        ("steps_per_temperature", steps_per_temperature, 1),
        ("frozen_levels", frozen_levels, 1),
        ("max_steps", max_steps, 0),
    ):
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}, or None")


class _Cover:
    """Current cover: subsets in a list with positions, and coverage counts."""

    def __init__(self, inst: Instance, selected: list[int]):
        self.inst = inst
        self.items: list[int] = []
        self.pos = [-1] * inst.m
        self.counts = [0] * inst.n
        for j in selected:
            self.add(j)

    def add(self, j: int) -> None:
        self.pos[j] = len(self.items)
        self.items.append(j)
        for e in self.inst.sets[j]:
            self.counts[e] += 1

    def remove(self, j: int) -> None:
        i = self.pos[j]
        last = self.items.pop()
        if last != j:
            self.items[i] = last
            self.pos[last] = i
        self.pos[j] = -1
        for e in self.inst.sets[j]:
            self.counts[e] -= 1


def _move(
    inst: Instance,
    cover: _Cover,
    is_fixed: list[bool],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Apply one move; return its log of (+1 added / -1 removed, subset).

    The cover must contain at least one non-fixed subset.
    """
    counts, pos, sets, items = cover.counts, cover.pos, inst.sets, cover.items
    j = items[int(rng.integers(len(items)))]
    while is_fixed[j]:
        j = items[int(rng.integers(len(items)))]
    cover.remove(j)
    log = [(-1, j)]

    uncovered = [e for e in sets[j] if counts[e] == 0]
    while uncovered:
        e = uncovered[int(rng.integers(len(uncovered)))]
        best_gain, ties = -1, []
        for k in inst.covering[e]:
            if k == j or pos[k] >= 0:
                continue
            gain = sum(1 for x in sets[k] if counts[x] == 0)
            if gain > best_gain:
                best_gain, ties = gain, [k]
            elif gain == best_gain:
                ties.append(k)
        k = ties[int(rng.integers(len(ties)))] if ties else j
        cover.add(k)
        log.append((1, k))
        uncovered = [x for x in uncovered if counts[x] == 0]

    touched = {q for sign, k in log if sign == 1
               for x in sets[k] for q in inst.covering[x] if pos[q] >= 0}
    for q in sorted(touched):
        if pos[q] >= 0 and all(counts[x] >= 2 for x in sets[q]):
            cover.remove(q)
            log.append((-1, q))
    return log


def _undo(cover: _Cover, log: list[tuple[int, int]]) -> None:
    for sign, k in reversed(log):
        if sign == 1:
            cover.remove(k)
        else:
            cover.add(k)


def _anneal(
    inst: Instance,
    initial: list[int],
    begin: float,
    deadline: float,
    rng: np.random.Generator,
    initial_temperature: float,
    cooling_rate: float,
    steps_per_temperature: int | None,
    min_acceptance: float,
    frozen_levels: int,
    max_steps: int | None,
) -> Result:
    cover = _Cover(inst, initial)
    # Only subset covering some element: removing it would force it back.
    is_fixed = [False] * inst.m
    for cols in inst.covering:
        if len(cols) == 1:
            is_fixed[cols[0]] = True
    # Fixed subsets are in every cover and are never removed.
    n_fixed = sum(is_fixed)
    best = sorted(initial)
    best_step, best_time = 0, time.perf_counter() - begin
    level_length = steps_per_temperature or max(1, len(initial))
    temperature = initial_temperature
    steps = accepted = accepted_in_level = frozen = restarts = 0

    while len(best) > 1:
        if time.perf_counter() >= deadline or (max_steps is not None and steps >= max_steps):
            break
        if len(cover.items) == n_fixed:
            break

        log = _move(inst, cover, is_fixed, rng)
        delta = sum(sign for sign, _ in log)
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            accepted += 1
            accepted_in_level += 1
            if len(cover.items) < len(best):
                best = sorted(cover.items)
                best_step, best_time = steps + 1, time.perf_counter() - begin
                frozen = 0
        else:
            _undo(cover, log)
        steps += 1

        if steps % level_length == 0:
            frozen = frozen + 1 if accepted_in_level < min_acceptance * level_length else 0
            accepted_in_level = 0
            temperature *= cooling_rate
            if frozen >= frozen_levels:
                temperature, frozen = initial_temperature, 0
                restarts += 1

    return Result(
        selected=best,
        size=len(best),
        status="feasible",
        timed_out=time.perf_counter() >= deadline,
        extra={
            "initial_size": len(initial),
            "steps": steps,
            "accepted_moves": accepted,
            "restarts": restarts,
            "best_step": best_step,
            "best_time": best_time,
        },
    )


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.95,
    steps_per_temperature: int | None = None,
    min_acceptance: float = 0.02,
    frozen_levels: int = 5,
    max_steps: int | None = None,
) -> Result:
    """Search from the greedy cover; a level lasts |C| moves by default."""
    begin = time.perf_counter()
    deadline = begin + time_limit
    _validate_parameters(time_limit, initial_temperature, cooling_rate,
                         steps_per_temperature, min_acceptance, frozen_levels, max_steps)

    if inst.n == 0:
        return Result(selected=[], size=0, status="optimal", lower_bound=0)
    initial = greedy_cover(inst, max(0.0, deadline - time.perf_counter()), rng)
    if initial.status == "none":
        return initial

    return _anneal(inst, initial.selected, begin, deadline, rng, initial_temperature,
                   cooling_rate, steps_per_temperature, min_acceptance, frozen_levels,
                   max_steps)