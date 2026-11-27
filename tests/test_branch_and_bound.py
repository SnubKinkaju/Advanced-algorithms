import time
from pathlib import Path

import numpy as np
import pytest

from setcover.bounds import counting_bound
from setcover.generator import generate
from setcover.instance import make_instance
from setcover.io import read_instance
from setcover.methods import branch_and_bound, highs_ref
from setcover.registry import METHODS
from setcover.result import verify

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def run(inst, time_limit=10.0, **params):
    return branch_and_bound.solve(inst, time_limit, np.random.default_rng(0), **params)


def test_small_instance():
    inst = make_instance("toy", 5, [[0, 1, 2], [2, 3], [3, 4], [0, 4], [1]])
    res = run(inst)
    verify(inst, res)
    assert res.status == "optimal"
    assert res.size == 2


def test_beats_greedy():
    # Greedy takes {0,1,2,3} then two other subsets: 3 in total.
    # The optimum is {0,1,4} + {2,3,5}: 2 subsets.
    inst = make_instance("trap", 6, [[0, 1, 2, 3], [0, 1, 4], [2, 3, 5]])
    res = run(inst)
    verify(inst, res)
    assert res.status == "optimal"
    assert res.size == 2


def test_uncoverable():
    inst = make_instance("bad", 3, [[0, 1]])
    res = run(inst)
    verify(inst, res)
    assert res.status == "none"
    assert res.extra["reason"] == "infeasible"


def test_zero_time_limit():
    inst = make_instance("trap", 6, [[0, 1, 2, 3], [0, 1, 4], [2, 3, 5]])
    res = run(inst, time_limit=0.0)
    verify(inst, res)
    assert res.status == "none"
    assert res.timed_out


def test_timeout_returns_valid_cover():
    inst = read_instance(BENCHMARKS / "clr" / "scpclr10.txt")
    res = run(inst, time_limit=1.0)
    verify(inst, res)
    assert res.timed_out
    assert res.status == "feasible"
    assert res.lower_bound <= res.size


def test_matches_highs_on_random_instances():
    rng = np.random.default_rng(42)
    for k in range(20):
        n, m = 12, 15
        sets = [[e for e in range(n) if rng.random() < 0.3] for _ in range(m)]
        for e in range(n):
            sets[rng.integers(m)].append(e)
        inst = make_instance(f"rand{k}", n, sets)
        res = run(inst)
        ref = highs_ref.solve(inst, 10.0, np.random.default_rng(0))
        verify(inst, res)
        assert res.status == "optimal"
        assert res.size == ref.size


def test_matches_brute_force_and_dp_with_and_without_lp():
    for seed in range(30):
        inst = generate(6 + seed % 12, 6 + seed % 14, [0.15, 0.25, 0.35][seed % 3], seed)
        bf = METHODS["brute_force"](inst, 30.0, np.random.default_rng(seed))
        dp = METHODS["dp_bottom_up"](inst, 30.0, np.random.default_rng(seed))
        for use_lp in (True, False):
            res = run(inst, use_lp=use_lp)
            verify(inst, res)
            assert res.status == "optimal"
            assert res.size == bf.size == dp.size


@pytest.mark.parametrize(
    "relative_path, optimum",
    [("sts/data.9", 5), ("sts/data.15", 9), ("sts/data.27", 18), ("e/scpe1.txt", 5)],
)
def test_benchmarks_solved_to_optimality(relative_path, optimum):
    inst = read_instance(BENCHMARKS / relative_path)
    res = run(inst, time_limit=60.0)
    verify(inst, res)
    assert res.status == "optimal"
    assert res.size == optimum


def test_starts_from_the_greedy_cover():
    inst = read_instance(BENCHMARKS / "cyc" / "scpcyc06.txt")
    start = METHODS["greedy"](inst, 10.0, np.random.default_rng(0))
    res = run(inst, time_limit=0.5)
    verify(inst, res)
    assert res.size <= start.size


def test_time_limit_respected_on_large_instance():
    inst = read_instance(BENCHMARKS / "cyc" / "scpcyc11.txt")
    begin = time.perf_counter()
    res = run(inst, time_limit=1.0)
    elapsed = time.perf_counter() - begin
    verify(inst, res)
    assert res.timed_out
    assert elapsed < 3.0


def test_time_limit_respected_without_lp_on_large_instance():
    inst = read_instance(BENCHMARKS / "sts" / "data.1215")
    begin = time.perf_counter()
    res = run(inst, time_limit=1.0, use_lp=False)
    elapsed = time.perf_counter() - begin
    verify(inst, res)
    assert res.timed_out
    assert elapsed < 2.0


@pytest.mark.parametrize("relative_path, bound", [("sts/data.1215", 405), ("sts/data.2187", 729)])
def test_time_limit_respected_with_lp_on_regular_instance(relative_path, bound):
    # Regular instance: the LP value is n/p, so HiGHS is not called.
    inst = read_instance(BENCHMARKS / relative_path)
    inst.masks  # built before the clock, as in the runner
    begin = time.perf_counter()
    res = run(inst, time_limit=1.0)
    elapsed = time.perf_counter() - begin
    verify(inst, res)
    assert res.timed_out
    assert res.lower_bound == res.extra["lp_bound"] == bound
    assert elapsed < 2.0


@pytest.mark.parametrize("relative_path", ["sts/data.1215", "cyc/scpcyc11.txt"])
def test_search_starts_with_lp_on_regular_instance(relative_path):
    # Before the shortcut, HiGHS took the budget and no node was explored.
    inst = read_instance(BENCHMARKS / relative_path)
    res = run(inst, time_limit=1.0)
    verify(inst, res)
    assert res.nodes > 0


def test_counting_bound_is_the_default_root_bound():
    inst = read_instance(BENCHMARKS / "sts" / "data.1215")
    res = run(inst, time_limit=1.0, use_lp=False)
    verify(inst, res)
    assert res.lower_bound == counting_bound(inst) == 405
    assert res.extra == {"lp_bound": None, "lp_time": 0.0}


def test_stops_when_the_search_reaches_the_root_bound():
    # Greedy gives 12 here, the LP bound is 10 and the search finds a cover of
    # size 10 after a few nodes. That cover is optimal, so the search must stop
    # and say so, instead of exploring the rest of the tree (about 22,000 nodes
    # before the fix) and returning "feasible" if the time limit is reached.
    inst = generate(50, 150, 0.08, seed=6)
    res = run(inst, time_limit=5.0)
    verify(inst, res)
    assert res.status == "optimal"
    assert res.size == res.lower_bound == 10
    assert res.nodes < 1000


def test_a_cover_equal_to_its_lower_bound_is_declared_optimal():
    inst = generate(50, 150, 0.08, seed=6)
    for limit in (0.05, 0.5, 5.0):
        res = run(inst, time_limit=limit)
        if res.size == res.lower_bound:
            assert res.status == "optimal"


def test_registry():
    assert METHODS["branch_and_bound"] is branch_and_bound.solve