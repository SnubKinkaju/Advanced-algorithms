
import time

import numpy as np

from setcover.instance import Instance
from setcover.result import Result


MAX_STATES = 1 << 20  # Maximum : 2^20 états


def solve(
    inst: Instance,
    time_limit: float,
    rng: np.random.Generator
) -> Result:

    deadline = time.perf_counter() + time_limit

    # 1. Nombre d'états
    n = inst.n
    total_states = 1 << n
    full_mask = total_states - 1

    # 2. Vérification des ressources
    if time.perf_counter() >= deadline:
        return Result(
            selected=[],
            size=0,
            status="none",
            timed_out=True
        )

    if total_states > MAX_STATES:
        return Result(
            selected=[],
            size=0,
            status="none",
            timed_out=False,
            extra={"reason": "memory_limit"}
        )

    # 3. Initialisation
    dp = [float("inf")] * total_states
    parent = [-1] * total_states
    chosen = [-1] * total_states

    dp[0] = 0

    # 4. Transitions DP
    for mask in range(total_states):

        if time.perf_counter() >= deadline:
            return Result(
                selected=[],
                size=0,
                status="none",
                timed_out=True
            )

        if dp[mask] == float("inf"):
            continue

        for j, subset_mask in enumerate(inst.masks):

            if time.perf_counter() >= deadline:
                return Result(
                    selected=[],
                    size=0,
                    status="none",
                    timed_out=True
                )

            new_mask = mask | subset_mask
            new_cost = dp[mask] + 1

            if new_cost < dp[new_mask]:
                dp[new_mask] = new_cost
                parent[new_mask] = mask
                chosen[new_mask] = j

    # 5. Aucune couverture possible
    if dp[full_mask] == float("inf"):
        return Result(
            selected=[],
            size=0,
            status="none",
            timed_out=False
        )

    # 6. Reconstruction
    selected_sets = []
    mask = full_mask

    while mask != 0:
        j = chosen[mask]
        selected_sets.append(j)
        mask = parent[mask]

    selected_sets.reverse()

    # 7. Résultat optimal
    return Result(
        selected=selected_sets,
        size=len(selected_sets),
        status="optimal",
        timed_out=False,
        lower_bound=len(selected_sets)
    )
