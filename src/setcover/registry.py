from setcover.methods import (
    branch_and_bound,
    branch_and_bound_v2,
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
    "branch_and_bound_v2": branch_and_bound_v2.solve,
    "brute_force": brute_force.solve,
    "dp_bottom_up": dp_bottom_up.solve,
    "dp_memoized": dp_memoized.solve,
    "greedy": greedy.solve,
    "highs": highs_ref.solve,
    "rwls": rwls.solve,
    "trivial": trivial.solve,
}
