from setcover.methods import highs_ref, trivial

METHODS = {
    "highs": highs_ref.solve,
    "trivial": trivial.solve,
}
