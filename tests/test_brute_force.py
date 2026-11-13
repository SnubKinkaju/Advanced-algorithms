from math import comb
from pathlib import Path

import numpy as np

from setcover.generator import generate
from setcover.instance import make_instance
from setcover.io import read_instance
from setcover.methods.brute_force import solve
from setcover.registry import METHODS
from setcover.result import verify

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def run_bf(inst, time_limit=30.0):
    result = solve(inst, time_limit, np.random.default_rng(0))
    verify(inst, result)
    return result


def test_toy_instance():
    result = run_bf(make_instance("toy", 4, [[0, 1], [1, 2], [2, 3]]))
    assert result.status == "optimal"
    assert result.size == 2
    assert set(result.selected) == {0, 2}


def test_empty_universe():
    result = run_bf(make_instance("empty", 0, []))
    assert result.status == "optimal"
    assert result.size == 0


def test_impossible():
    result = run_bf(make_instance("impossible", 3, [[0], [1]]))
    assert result.status == "none"
    assert result.extra["reason"] == "infeasible"


def test_matches_highs_and_dp_on_generated_instances():
    for seed in range(50):
        inst = generate(4 + seed % 13, 3 + seed % 16, [0.1, 0.2, 0.35][seed % 3], seed)
        bf = run_bf(inst)
        ref = METHODS["highs"](inst, 30.0, np.random.default_rng(seed))
        dp = METHODS["dp_bottom_up"](inst, 30.0, np.random.default_rng(seed))
        assert bf.status == "optimal"
        assert ref.status == "optimal"
        assert bf.size == ref.size == dp.size


def test_number_of_combinations_examined():
    for seed in range(20):
        inst = generate(10, 12 + seed % 6, 0.2, seed)
        result = run_bf(inst)
        k, m = result.size, inst.m
        tried_before = sum(comb(m, i) for i in range(k))
        assert tried_before < result.extra["checked"] <= tried_before + comb(m, k)


def test_sts9_and_sts15():
    for name, optimum in (("data.9", 5), ("data.15", 9)):
        result = run_bf(read_instance(BENCHMARKS / "sts" / name))
        assert result.status == "optimal"
        assert result.size == optimum


def test_time_limit_gives_a_valid_lower_bound():
    inst = read_instance(BENCHMARKS / "e" / "scpe1.txt")
    result = run_bf(inst, time_limit=0.5)
    assert result.status == "none"
    assert result.timed_out
    assert 1 <= result.lower_bound <= 5


def test_registry():
    assert METHODS["brute_force"] is solve
