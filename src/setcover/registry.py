from setcover.methods import highs_ref, trivial, dp_bottom_up, rwls, branch_and_bound

METHODS = {
    "highs": highs_ref.solve,
    "trivial": trivial.solve,
    "dp_bottom_up": dp_bottom_up.solve,
    "rwls": rwls.solve,
    "branch_and_bound": branch_and_bound.solve,
}
