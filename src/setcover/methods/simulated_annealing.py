"""Simulated annealing on feasible covers, using single-subset additions/removals.

The energy is the number of selected subsets. A worse move is accepted with
probability exp(-delta / temperature); the temperature decreases geometrically.
Based on Kirkpatrick et al. (1983), doi:10.1126/science.220.4598.671.
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
    min_temperature: float,
    steps_per_temperature: int | None,
    max_steps: int | None,
) -> None:
    if not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be finite and non-negative")
    if not math.isfinite(initial_temperature) or initial_temperature <= 0:
        raise ValueError("initial_temperature must be finite and positive")
    if not math.isfinite(cooling_rate) or not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1 (exclusive)")
    if not math.isfinite(min_temperature) or not 0 < min_temperature < initial_temperature:
        raise ValueError("min_temperature must be positive and below initial_temperature")
    for name, value, minimum in (
        ("steps_per_temperature", steps_per_temperature, 1),
        ("max_steps", max_steps, 0),
    ):
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}, or None")


def _coverage_counts(inst: Instance, selected: set[int], deadline: float) -> list[int] | None:
    """Count how many selected subsets cover each element."""
    counts = [0] * inst.n
    for j in sorted(selected):
        for position, e in enumerate(inst.sets[j]):
            if position % 1024 == 0 and time.perf_counter() >= deadline:
                return None
            counts[e] += 1
    return counts


def _try_move(
    inst: Instance,
    selected: set[int],
    counts: list[int],
    j: int,
    temperature: float,
    rng: np.random.Generator,
) -> bool:
    """Toggle one subset, preserving coverage and applying the Metropolis rule."""
    if j in selected:
        # Reject any removal that would leave an element uncovered.
        if any(counts[e] == 1 for e in inst.sets[j]):
            return False
        selected.remove(j)
        change = -1
    else:
        # Adding a subset increases the energy by one.
        if rng.random() >= math.exp(-1.0 / temperature):
            return False
        selected.add(j)
        change = 1

    for e in inst.sets[j]:
        counts[e] += change
    return True


def _anneal(
    inst: Instance,
    initial: list[int],
    deadline: float,
    rng: np.random.Generator,
    temperature: float,
    cooling_rate: float,
    min_temperature: float,
    steps_per_temperature: int,
    max_steps: int | None,
) -> Result:
    selected = set(initial)
    best = sorted(selected)
    counts = _coverage_counts(inst, selected, deadline)
    steps = accepted = 0

    while counts is not None and temperature > min_temperature and len(best) > 1:
        if time.perf_counter() >= deadline or (max_steps is not None and steps >= max_steps):
            break
        j = int(rng.integers(inst.m))
        if _try_move(inst, selected, counts, j, temperature, rng):
            accepted += 1
            if len(selected) < len(best):
                best = sorted(selected)
        steps += 1
        # Keep each temperature for a fixed number of attempted moves.
        if steps % steps_per_temperature == 0:
            temperature *= cooling_rate

    return Result(
        selected=best,
        size=len(best),
        status="feasible",
        timed_out=time.perf_counter() >= deadline,
        extra={
            "initial_size": len(initial),
            "steps": steps,
            "accepted_moves": accepted,
            "temperature": temperature,
        },
    )


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.01,
    steps_per_temperature: int | None = None,
    max_steps: int | None = None,
) -> Result:
    """Search from a greedy cover; by default, try m moves per temperature."""
    deadline = time.perf_counter() + time_limit
    _validate_parameters(
        time_limit, initial_temperature, cooling_rate, min_temperature,
        steps_per_temperature, max_steps,
    )

    # Handle empty instances and let the shared greedy method build a cover.
    if inst.n == 0:
        return Result(selected=[], size=0, status="optimal", lower_bound=0)
    initial = greedy_cover(inst, max(0.0, deadline - time.perf_counter()), rng)
    if initial.status == "none":
        return initial

    # The same deadline includes initialization and the whole search.
    return _anneal(
        inst, initial.selected, deadline, rng, initial_temperature,
        cooling_rate, min_temperature,
        steps_per_temperature if steps_per_temperature is not None else inst.m,
        max_steps,
    )
