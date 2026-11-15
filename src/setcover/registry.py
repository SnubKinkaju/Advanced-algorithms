from setcover.methods import brute_force, dp_bottom_up, highs_ref, rwls, trivial

METHODS = {
    "brute_force": brute_force.solve,
    "dp_bottom_up": dp_bottom_up.solve,
    "highs": highs_ref.solve,
    "rwls": rwls.solve,
    "trivial": trivial.solve,
}
