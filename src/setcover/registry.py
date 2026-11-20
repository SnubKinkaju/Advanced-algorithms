from functools import partial

from setcover.methods import (
    branch_and_bound,
    brute_force,
    dp_bottom_up,
    dp_memoized,
    greedy,
    highs_ref,
    rwls,
    trivial,
)

METHODS = {
    "branch_and_bound": branch_and_bound.solve,
    "brute_force": brute_force.solve,
    "dp_bottom_up": dp_bottom_up.solve,
    "dp_memoized": dp_memoized.solve,
    "greedy": greedy.solve,
    # Greedy variants: the same function with one option switched on.
    "greedy_clean": partial(greedy.solve, remove_redundant=True),
    "greedy_random": partial(greedy.solve, random_ties=True),
    "highs": highs_ref.solve,
    "rwls": rwls.solve,
    "trivial": trivial.solve,
}
