from setcover.methods import (
    branch_and_bound,
    brute_force,
    dp_bottom_up,
    dp_memoized,
    greedy,
    highs_ref,
    rwls,
    simulated_annealing,
    trivial,
)

METHODS = {
    "branch_and_bound": branch_and_bound.solve,
    "brute_force": brute_force.solve,
    "dp_bottom_up": dp_bottom_up.solve,
    "dp_memoized": dp_memoized.solve,
    "greedy": greedy.solve,
    "highs": highs_ref.solve,
    "rwls": rwls.solve,
    "simulated_annealing": simulated_annealing.solve,
    "trivial": trivial.solve,
}
