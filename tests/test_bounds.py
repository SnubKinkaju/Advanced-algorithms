import time
from pathlib import Path

import numpy as np
import pytest

import setcover.bounds as bounds
from setcover.bounds import counting_bound, lp_bound, lp_relaxation, regular_lp_value
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


def test_lp_bound_with_enough_time_is_unchanged():
    inst = read_instance(BENCHMARKS / "e" / "scpe1.txt")
    assert lp_bound(inst, time_limit=10.0) == lp_bound(inst) == 4


def test_lp_relaxation_respects_time_limit():
    inst = read_instance(BENCHMARKS / "cyc" / "scpcyc11.txt")
    start = time.perf_counter()
    assert lp_relaxation(inst, time_limit=1.0) is None
    assert time.perf_counter() - start < 3.0


def test_regular_lp_value_on_toy_instances():
    # A 4-cycle: every subset has 2 elements and every element is in 2 subsets.
    square = make_instance("square", 4, [[0, 1], [1, 2], [2, 3], [3, 0]])
    assert regular_lp_value(square) == 2.0
    assert lp_relaxation(square) == pytest.approx(2.0)
    assert regular_lp_value(TOY) is None
    assert regular_lp_value(make_instance("empty", 0, [[]])) is None


@pytest.mark.parametrize("relative_path", [
    "sts/data.9", "sts/data.15", "sts/data.27", "sts/data.45", "sts/data.81",
    "sts/data.135", "sts/data.243", "cyc/scpcyc06.txt", "cyc/scpcyc07.txt", "cyc/scpcyc08.txt",
])
def test_regular_lp_value_matches_highs(relative_path):
    inst = read_instance(BENCHMARKS / relative_path)
    value = regular_lp_value(inst)
    assert value is not None
    assert value == pytest.approx(lp_relaxation(inst), abs=1e-6)
    assert lp_bound(inst) == counting_bound(inst)


@pytest.mark.parametrize("relative_path, expected", [("e/scpe1.txt", 4), ("clr/scpclr10.txt", 21)])
def test_irregular_instances_still_use_highs(relative_path, expected):
    inst = read_instance(BENCHMARKS / relative_path)
    assert regular_lp_value(inst) is None
    assert lp_bound(inst) == expected > counting_bound(inst)


def test_lp_bound_does_not_call_highs_on_regular_instances(monkeypatch):
    def no_solver(*args, **kwargs):
        raise AssertionError("HiGHS must not be called on a regular instance")

    monkeypatch.setattr(bounds, "lp_relaxation", no_solver)
    inst = read_instance(BENCHMARKS / "sts" / "data.2187")
    start = time.perf_counter()
    assert lp_bound(inst, time_limit=0.0) == 729
    assert time.perf_counter() - start < 0.5


def test_counting_bound_shortcut_matches_the_general_case():
    instances = [read_instance(BENCHMARKS / "e" / "scpe1.txt"), read_instance(BENCHMARKS / "sts" / "data.45")]
    instances += [generate(15, 25, 0.2, seed=seed) for seed in range(5)]
    for inst in instances:
        assert counting_bound(inst) == counting_bound(inst, uncovered=set(range(inst.n)))