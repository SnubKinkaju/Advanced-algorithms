
import time

import numpy as np

from setcover.instance import Instance
from setcover.result import Result


MAX_RECURSION_DEPTH = 500
MAX_MEMO_STATES = 1 << 20


class DPTimeout(Exception):
    pass


class DPResourceLimit(Exception):
    pass


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator
) -> Result:

    deadline = time.perf_counter() + time_limit

    n = inst.n
    full_mask = (1 << n) - 1

    memo = {}
    choice = {}

    # Vérifier la limite de temps initiale
    if time.perf_counter() >= deadline:
        return Result(
            selected=[],
            size=0,
            status="none",
            timed_out=True
        )

    # Éviter une récursion trop profonde
    if n > MAX_RECURSION_DEPTH:
        return Result(
            selected=[],
            size=0,
            status="none",
            timed_out=False,
            extra={"reason": "recursion_limit"}
        )

    def dp(mask):

        if time.perf_counter() >= deadline:
            raise DPTimeout()

        if mask == full_mask:
            return 0

        if mask in memo:
            return memo[mask]

        if len(memo) >= MAX_MEMO_STATES:
            raise DPResourceLimit()

        best = float("inf")

        for j, subset_mask in enumerate(inst.masks):

            if time.perf_counter() >= deadline:
                raise DPTimeout()

            new_mask = mask | subset_mask

            if new_mask == mask:
                continue

            cost = 1 + dp(new_mask)

            if cost < best:
                best = cost
                choice[mask] = j

        memo[mask] = best
        return best

    # Exécuter la recherche récursive
    try:
        optimal_cost = dp(0)

    except DPTimeout:
        return Result(
            selected=[],
            size=0,
            status="none",
            timed_out=True
        )

    except DPResourceLimit:
        return Result(
            selected=[],
            size=0,
            status="none",
            timed_out=False,
            extra={"reason": "memory_limit"}
        )

    # Vérifier si une couverture existe
    if optimal_cost == float("inf"):
        return Result(
            selected=[],
            size=0,
            status="none",
            timed_out=False
        )

    # Reconstruire la solution optimale
    selected_sets = []
    mask = 0

    while mask != full_mask:

        j = choice[mask]
        selected_sets.append(j)

        mask = mask | inst.masks[j]

    return Result(
        selected=selected_sets,
        size=len(selected_sets),
        status="optimal",
        timed_out=False,
        lower_bound=len(selected_sets)
    )
