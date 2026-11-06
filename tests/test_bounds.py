from pathlib import Path

import numpy as np
import pytest

from setcover.bounds import counting_bound, lp_bound, lp_relaxation
from setcover.generator import generate
from setcover.instance import make_instance
from setcover.io import read_instance
from setcover.registry import METHODS

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
TOY = make_instance("toy", 4, [[0, 1], [1, 2], [2, 3]])


def test_bounds_on_toy_instance():
    assert lp_bound(TOY) == 2
    assert counting_bound(TOY) == 2


def test_counting_bound_on_partial_cover():
    assert counting_bound(TOY, uncovered=set()) == 0
    assert counting_bound(TOY, uncovered={3}) == 1
    assert counting_bound(TOY, uncovered={0, 3}) == 2


def test_lp_relaxation_of_e1():
    inst = read_instance(BENCHMARKS / "e" / "scpe1.txt")
    assert lp_relaxation(inst) == pytest.approx(3.48, abs=0.01)
    assert lp_bound(inst) == 4


def test_bounds_never_exceed_the_optimum():
    for seed in range(10):
        inst = generate(12, 20, 0.2, seed=seed)
        best = METHODS["highs"](inst, 10.0, np.random.default_rng(seed))
        assert best.status == "optimal"
        assert lp_bound(inst) <= best.size
        assert counting_bound(inst) <= best.size
