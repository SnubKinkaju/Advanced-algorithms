from pathlib import Path

import numpy as np
import pytest

from setcover.generator import generate
from setcover.instance import make_instance
from setcover.io import read_instance
from setcover.methods.greedy import solve
from setcover.registry import METHODS
from setcover.result import verify
from setcover.runner import run_one

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
BENCHMARK_FILES = sorted(f for f in BENCHMARKS.glob("*/*") if f.name != "SOURCE.md")
TRAP = make_instance("trap", 6, [[1, 2, 3, 4], [0, 1, 2], [3, 4, 5]])


def run_greedy(inst, seed=0, **params):
    result = solve(inst, 30.0, np.random.default_rng(seed), **params)
    verify(inst, result)
    return result


def harmonic(d):
    return sum(1 / i for i in range(1, d + 1))


def test_trap_without_cleaning():
    result = run_greedy(TRAP)
    assert result.status == "feasible"
    assert result.selected == [0, 1, 2]
    assert result.extra["removed"] == 0


def test_trap_with_cleaning_reaches_optimum():
    result = run_greedy(TRAP, remove_redundant=True)
    assert result.selected == [1, 2]
    assert result.extra["removed"] == 1


def test_one_set_covers_everything():
    result = run_greedy(make_instance("one", 4, [[0, 1], [1, 2], [0, 1, 2, 3]]))
    assert result.selected == [2]


def test_empty_universe():
    result = run_greedy(make_instance("empty", 0, []))
    assert result.size == 0


def test_impossible():
    result = run_greedy(make_instance("impossible", 3, [[0], [1]]))
    assert result.status == "none"
    assert result.extra["reason"] == "infeasible"


@pytest.mark.parametrize("path", BENCHMARK_FILES, ids=lambda p: p.name)
def test_valid_cover_on_all_benchmarks(path):
    inst = read_instance(path)
    plain = run_greedy(inst)
    cleaned = run_greedy(inst, remove_redundant=True)
    assert cleaned.size <= plain.size


@pytest.mark.parametrize(
    "relative_path, expected",
    [("e/scpe1.txt", 5), ("e/scpe4.txt", 6), ("cyc/scpcyc06.txt", 60), ("clr/scpclr10.txt", 33), ("sts/data.27", 19)],
)
def test_known_greedy_values(relative_path, expected):
    assert run_greedy(read_instance(BENCHMARKS / relative_path)).size == expected


def test_never_below_optimum_and_within_guarantee():
    for seed in range(50):
        inst = generate(10 + seed % 30, 15 + seed % 40, [0.1, 0.2, 0.3][seed % 3], seed)
        optimum = METHODS["highs"](inst, 30.0, np.random.default_rng(seed))
        assert optimum.status == "optimal"
        d = max(len(s) for s in inst.sets)
        for params in ({}, {"remove_redundant": True}, {"random_ties": True}):
            result = run_greedy(inst, seed=seed, **params)
            assert optimum.size <= result.size <= harmonic(d) * optimum.size + 1e-9


def test_random_ties_same_seed_same_result():
    inst = generate(40, 80, 0.1, seed=3)
    first = run_greedy(inst, seed=7, random_ties=True)
    second = run_greedy(inst, seed=7, random_ties=True)
    assert first.selected == second.selected


def test_random_ties_explores_different_covers():
    inst = generate(40, 80, 0.1, seed=3)
    sizes = {run_greedy(inst, seed=s, random_ties=True).size for s in range(30)}
    assert len(sizes) > 1


def test_default_is_deterministic():
    inst = generate(40, 80, 0.1, seed=3)
    assert run_greedy(inst, seed=1).selected == run_greedy(inst, seed=2).selected


def test_time_limit_zero():
    result = solve(TRAP, 0.0, np.random.default_rng(0))
    assert result.status == "none"
    assert result.timed_out


def test_run_one_keeps_extra():
    record = run_one("greedy", BENCHMARKS / "cyc" / "scpcyc07.txt", seed=0, time_limit=10.0, remove_redundant=True)
    assert record["valid"]
    assert record["size"] == 144
    assert record["extra"]["removed"] == 4


def test_registry():
    assert METHODS["greedy"] is solve