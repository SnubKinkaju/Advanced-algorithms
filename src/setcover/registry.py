from setcover.methods import highs_ref, trivial, dp_bottom_up, rwls

METHODS = {
    "highs": highs_ref.solve,
    "trivial": trivial.solve,
    "dp_bottom_up": dp_bottom_up.solve,
    "rwls": rwls.solve,
}
