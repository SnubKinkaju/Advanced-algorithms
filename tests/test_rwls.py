import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from setcover.generator import generate
from setcover.instance import make_instance
from setcover.io import read_instance
from setcover.methods import greedy, rwls
from setcover.methods.rwls import solve
from setcover.registry import METHODS
from setcover.result import verify
from setcover.runner import run_one

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def run_rwls(inst, time_limit=30.0, seed=0, max_steps=None):
    result = solve(inst, time_limit, np.random.default_rng(seed), max_steps=max_steps)
    verify(inst, result)
    return result


def test_toy_instance():
    result = run_rwls(make_instance("toy", 4, [[0, 1], [1, 2], [2, 3]]))
    assert result.size == 2
    assert set(result.selected) == {0, 2}


def test_empty_universe():
    result = run_rwls(make_instance("empty", 0, []))
    assert result.status == "optimal"
    assert result.size == 0


@pytest.mark.parametrize(
    "relative_path, optimum",
    [("e/scpe1.txt", 5), ("sts/data.27", 18), ("cyc/scpcyc06.txt", 60), ("clr/scpclr10.txt", None)],
)
def test_valid_cover_on_benchmarks(relative_path, optimum):
    result = run_rwls(read_instance(BENCHMARKS / relative_path), max_steps=2000)
    assert result.status == "feasible"
    if optimum is not None:
        assert result.size >= optimum


def test_never_below_highs_optimum():
    for seed in range(20):
        inst = generate(15 + seed, 25 + seed, 0.15, seed)
        best = METHODS["highs"](inst, 30.0, np.random.default_rng(seed))
        result = run_rwls(inst, seed=seed, max_steps=500)
        assert best.status == "optimal"
        assert result.size >= best.size


def test_same_seed_and_steps_give_same_result():
    inst = generate(60, 120, 0.05, seed=3)
    first = run_rwls(inst, seed=7, max_steps=1000)
    second = run_rwls(inst, seed=7, max_steps=1000)
    assert first.selected == second.selected


def test_max_steps_is_respected():
    inst = generate(60, 120, 0.05, seed=3)
    result = run_rwls(inst, max_steps=300)
    assert result.extra["steps"] <= 300


def test_time_limit_is_respected():
    record = run_one("rwls", BENCHMARKS / "cyc" / "scpcyc11.txt", seed=0, time_limit=1.0)
    assert record["valid"]
    assert record["timed_out"]
    assert record["time"] < 3.0


@pytest.mark.parametrize("time_limit, max_steps", [(-1.0, None), (float("inf"), None), (10.0, -5)])
def test_invalid_parameters_are_refused(time_limit, max_steps):
    inst = make_instance("toy", 4, [[0, 1], [1, 2], [2, 3]])
    with pytest.raises(ValueError):
        solve(inst, time_limit, np.random.default_rng(0), max_steps=max_steps)


def test_incremental_scores_are_exact():
    inst = read_instance(BENCHMARKS / "clr" / "scpclr10.txt")
    rng = np.random.default_rng(0)
    state = rwls._SearchState(inst)
    assert rwls._initial_cover(inst, state, time.perf_counter() + 60, rng)
    tabu = deque(maxlen=2)
    for step in range(1, 301):
        if not state.uncovered:
            removable = sorted(state.selected - state.fixed)
            rwls._remove_subset(inst, state, rwls._choose_subset(removable, state, rng, use_age=False))
        else:
            rwls._exchange_subset(inst, state, tabu, step, rng)
        expected = []
        for j, s in enumerate(inst.sets):
            if j in state.selected:
                expected.append(-sum(state.weights[e] for e in s if state.cover_count[e] == 1))
            else:
                expected.append(sum(state.weights[e] for e in s if state.cover_count[e] == 0))
        assert state.scores == expected


def test_registry():
    assert METHODS["rwls"] is solve


def test_initial_cover_is_the_random_tie_greedy():
    # Same rule as the article: highest score, ties at random.
    inst = read_instance(BENCHMARKS / "cyc" / "scpcyc07.txt")
    for seed in range(5):
        state = rwls._SearchState(inst)
        assert rwls._initial_cover(inst, state, time.perf_counter() + 60, np.random.default_rng(seed))
        expected = greedy.solve(inst, 60.0, np.random.default_rng(seed), random_ties=True)
        assert sorted(state.selected) == sorted(expected.selected)
        assert not state.uncovered


def test_cover_found_quickly_on_the_largest_cyc():
    record = run_one("rwls", BENCHMARKS / "cyc" / "scpcyc11.txt", seed=0, time_limit=2.0)
    assert record["valid"]
    assert record["size"] > 0


def test_best_time_and_step_are_recorded():
    inst = read_instance(BENCHMARKS / "sts" / "data.27")
    begin = time.perf_counter()
    result = solve(inst, 1.0, np.random.default_rng(0))
    elapsed = time.perf_counter() - begin
    verify(inst, result)
    assert 0 <= result.extra["best_time"] <= elapsed
    assert 0 <= result.extra["best_step"] <= result.extra["steps"]


def test_indexed_set():
    s = rwls._IndexedSet(10, [3, 1, 4])
    s.remove(3)
    s.add(7)
    s.remove(7)
    s.add(9)
    assert sorted(s) == [1, 4, 9]
    assert len(s) == 3
    assert 4 in s and 3 not in s and 7 not in s
    for i, x in enumerate(s.items):
        assert s.pos[x] == i
    assert s - {4} == {1, 9}


def test_load_cover_matches_successive_additions():
    inst = read_instance(BENCHMARKS / "sts" / "data.81")
    cover = greedy.solve(inst, 60.0, np.random.default_rng(3), random_ties=True).selected
    loaded = rwls._SearchState(inst)
    rwls._load_cover(inst, loaded, cover)
    added = rwls._SearchState(inst)
    for j in cover:
        rwls._add_subset(inst, added, j)
    assert sorted(loaded.selected) == sorted(added.selected)
    assert loaded.cover_count == added.cover_count
    assert loaded.scores == added.scores
    assert loaded.can_add == added.can_add
    assert sorted(loaded.uncovered) == sorted(added.uncovered)