from pathlib import Path

import numpy as np

from setcover.generator import generate
from setcover.instance import make_instance
from setcover.io import read_instance
from setcover.methods.dp_bottom_up import solve
from setcover.registry import METHODS
from setcover.result import verify

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def run_dp(n, sets, time_limit=10.0):
    inst = make_instance("test", n, sets)
    result = solve(inst, time_limit, np.random.default_rng(0))
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
    assert set(result.selected) == {0, 1, 2}


def test_impossible():
    result = run_dp(3, [[0], [1]])
    assert result.status == "none"
    assert result.selected == []
    assert result.timed_out is False


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
    assert "dp_bottom_up" in METHODS
    assert METHODS["dp_bottom_up"] is solve


def test_matches_highs_on_generated_instances():
    for seed in range(50):
        inst = generate(4 + seed % 11, 3 + seed % 25, [0.1, 0.2, 0.35][seed % 3], seed)
        dp = solve(inst, 30.0, np.random.default_rng(seed))
        ref = METHODS["highs"](inst, 30.0, np.random.default_rng(seed))
        verify(inst, dp)
        assert dp.status == "optimal"
        assert ref.status == "optimal"
        assert dp.size == ref.size


def test_sts9_benchmark():
    inst = read_instance(BENCHMARKS / "sts" / "data.9")
    result = solve(inst, 30.0, np.random.default_rng(0))
    verify(inst, result)
    assert result.status == "optimal"
    assert result.size == 5


def test_too_many_elements_is_refused():
    inst = generate(21, 10, 0.3, seed=0)
    result = solve(inst, 10.0, np.random.default_rng(0))
    assert result.status == "none"
    assert result.timed_out is False
    assert result.extra["reason"] == "memory_limit"