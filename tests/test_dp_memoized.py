import time
from pathlib import Path

import numpy as np

from setcover.generator import generate
from setcover.instance import make_instance
from setcover.io import read_instance
from setcover.methods.dp_memoized import solve
from setcover.registry import METHODS
from setcover.result import verify

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def run_dp(n, sets, time_limit=5.0, **params):
    inst = make_instance("test", n, sets)
    result = solve(inst, time_limit, np.random.default_rng(42), **params)
    verify(inst, result)
    return result


def test_simple():
    result = run_dp(4, [[0, 1], [1, 2], [2, 3]])
    assert result.status == "optimal"
    assert result.size == 2
    assert set(result.selected) == {0, 2}


def test_one_set():
    result = run_dp(4, [[0, 1], [1, 2], [0, 1, 2, 3]])
    assert result.status == "optimal"
    assert result.size == 1
    assert result.selected == [2]


def test_all_sets_needed():
    result = run_dp(3, [[0], [1], [2]])
    assert result.status == "optimal"
    assert result.size == 3


def test_impossible():
    result = run_dp(3, [[0], [1]])
    assert result.status == "none"
    assert result.timed_out is False
    assert result.extra["reason"] == "infeasible"


def test_empty_universe():
    result = run_dp(0, [])
    assert result.status == "optimal"
    assert result.size == 0
    assert result.selected == []


def test_timeout():
    result = run_dp(4, [[0, 1], [1, 2], [2, 3]], time_limit=0.0)
    assert result.status == "none"
    assert result.timed_out is True


def test_registry():
    assert "dp_memoized" in METHODS
    assert METHODS["dp_memoized"] is solve


def test_matches_highs_on_generated_instances():
    for seed in range(50):
        inst = generate(4 + seed % 15, 3 + seed % 25, [0.1, 0.2, 0.35][seed % 3], seed)
        dp = solve(inst, 30.0, np.random.default_rng(seed))
        ref = METHODS["highs"](inst, 30.0, np.random.default_rng(seed))
        verify(inst, dp)
        assert dp.status == "optimal"
        assert ref.status == "optimal"
        assert dp.size == ref.size


def test_matches_bottom_up():
    for seed in range(20):
        inst = generate(6 + seed % 10, 5 + seed % 20, 0.25, seed)
        top_down = solve(inst, 30.0, np.random.default_rng(0))
        bottom_up = METHODS["dp_bottom_up"](inst, 30.0, np.random.default_rng(0))
        assert top_down.size == bottom_up.size


def test_sts9_benchmark():
    inst = read_instance(BENCHMARKS / "sts" / "data.9")
    result = solve(inst, 30.0, np.random.default_rng(0))
    verify(inst, result)
    assert result.status == "optimal"
    assert result.size == 5


def test_memory_limit():
    # The search needs more than 4 states, so it stops on the memo size.
    inst = generate(12, 20, 0.2, seed=0)
    result = solve(inst, 30.0, np.random.default_rng(0), max_states=4)
    verify(inst, result)
    assert result.status == "feasible"
    assert result.timed_out is False
    assert result.extra["reason"] == "memory_limit"


def test_too_deep_instance_refused_at_once():
    inst = generate(30, 10, 0.3, seed=0)
    begin = time.perf_counter()
    result = solve(inst, 30.0, np.random.default_rng(0), max_depth=20)
    assert time.perf_counter() - begin < 0.1
    assert result.status == "none"
    assert result.extra["reason"] == "too_large"


def test_time_limit_respected():
    inst = read_instance(BENCHMARKS / "sts" / "data.27")
    begin = time.perf_counter()
    result = solve(inst, 1.0, np.random.default_rng(0))
    elapsed = time.perf_counter() - begin
    verify(inst, result)
    assert result.status == "feasible"
    assert result.timed_out is True
    assert elapsed < 1.2

ONE_BIG_SET = make_instance("one_big_set", 20, [list(range(20))] + [[e] for e in range(20)])


def test_interrupted_search_keeps_the_cover_found():
    # The cover [0] is found at once; the full search takes seconds.
    result = solve(ONE_BIG_SET, 0.05, np.random.default_rng(0))
    verify(ONE_BIG_SET, result)
    assert result.status == "feasible"
    assert result.selected == [0]
    assert result.timed_out is True


def test_memory_limit_keeps_the_cover_found():
    result = solve(ONE_BIG_SET, 30.0, np.random.default_rng(0), max_states=100)
    verify(ONE_BIG_SET, result)
    assert result.status == "feasible"
    assert result.selected == [0]
    assert result.extra["reason"] == "memory_limit"


def test_interrupted_results_are_valid_covers():
    for seed in range(10):
        inst = generate(18, 30, 0.15, seed)
        result = solve(inst, 0.01, np.random.default_rng(0))
        verify(inst, result)
        assert result.status in ("none", "feasible", "optimal")