from setcover.methods import highs_ref, trivial, dp_bottom_up, dp_memoized

METHODS = {
    "highs": highs_ref.solve,
    "trivial": trivial.solve,
    "dp_bottom_up": dp_bottom_up.solve,
    "dp_memoized":dp_memoized.solve,
}
