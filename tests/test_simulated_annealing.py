import copy
import math
import time
from pathlib import Path

import numpy as np
import pytest

from setcover.generator import generate
from setcover.instance import make_instance
from setcover.methods import brute_force, greedy, simulated_annealing as sa
from setcover.registry import METHODS
from setcover.result import Result, verify
from setcover.runner import run_one

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
TRAP = make_instance(
    "trap", 6, [[0, 1, 2, 3], [0, 4], [2, 5], [0, 1, 4], [2, 3, 5]]
)


def run_sa(inst, seed=0, time_limit=10.0, **params):
    result = sa.solve(inst, time_limit, np.random.default_rng(seed), **params)
    verify(inst, result)
    return result


def recount(inst, items):
    return [sum(e in inst.sets[j] for j in items) for e in range(inst.n)]


def test_escapes_a_cover_that_cannot_be_improved_by_removals_alone():
    initial = greedy.solve(TRAP, 10.0, np.random.default_rng(0))
    assert initial.selected == [0, 1, 2]
    for removed in initial.selected:
        covered = {e for j in initial.selected if j != removed for e in TRAP.sets[j]}
        assert len(covered) < TRAP.n

    result = run_sa(TRAP, max_steps=200)
    assert result.selected == [3, 4]
    assert result.extra["initial_size"] == 3


def test_one_move_removes_repairs_and_cleans():
    # Removing 0 uncovers 1 and 3; the repair adds 3 and 4, which make 1 and 2 redundant.
    cover = sa._Cover(TRAP, [0, 1, 2])
    log = sa._move(TRAP, cover, [False, True, True, False, False], np.random.default_rng(0))
    assert sorted(cover.items) == [3, 4]
    assert sum(sign for sign, _ in log) == -1
    assert cover.counts == recount(TRAP, cover.items)


def test_moves_keep_a_cover_and_undo_restores_the_state():
    inst = generate(25, 40, 0.15, seed=4)
    cover = sa._Cover(inst, greedy.solve(inst, 10.0, np.random.default_rng(0)).selected)
    rng = np.random.default_rng(3)
    for step in range(300):
        before_items, before_counts = sorted(cover.items), list(cover.counts)
        log = sa._move(inst, cover, [False] * inst.m, rng)
        assert all(c > 0 for c in cover.counts)
        assert cover.counts == recount(inst, cover.items)
        assert all(cover.pos[j] == i for i, j in enumerate(cover.items))
        if step % 2:
            sa._undo(cover, log)
            assert sorted(cover.items) == before_items
            assert cover.counts == before_counts


def test_fixed_subsets_are_never_removed():
    # Subset 0 is the only one covering element 0.
    inst = make_instance("fixed", 4, [[0, 1], [1, 2], [2, 3], [1, 2, 3]])
    result = run_sa(inst, max_steps=300)
    assert 0 in result.selected


def test_best_time_and_step_are_recorded():
    inst = generate(40, 80, 0.1, seed=1)
    begin = time.perf_counter()
    result = run_sa(inst, time_limit=0.5)
    elapsed = time.perf_counter() - begin
    assert 0 <= result.extra["best_time"] <= elapsed
    assert 0 <= result.extra["best_step"] <= result.extra["steps"]


def test_frozen_search_is_restarted():
    # A tiny instance freezes quickly; the search must go on until max_steps.
    result = run_sa(TRAP, max_steps=2000, steps_per_temperature=4)
    assert result.extra["steps"] == 2000
    assert result.extra["restarts"] >= 1


def test_initialization_uses_the_same_time_budget(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(sa.time, "perf_counter", lambda: clock[0])

    def slow_greedy(inst, time_limit, rng):
        assert time_limit == 1.0
        clock[0] = 2.0
        return Result(selected=[0, 1, 2], size=3, status="feasible")

    monkeypatch.setattr(sa, "greedy_cover", slow_greedy)
    result = run_sa(TRAP, time_limit=1.0)
    assert result.size == 3
    assert result.timed_out
    assert result.extra["steps"] == 0


def test_empty_universe():
    result = run_sa(make_instance("empty", 0, []), time_limit=0.0)
    assert result.status == "optimal"
    assert result.size == result.lower_bound == 0


def test_impossible_instance():
    result = run_sa(make_instance("impossible", 3, [[0], [1]]))
    assert result.status == "none"
    assert result.extra["reason"] == "infeasible"


def test_zero_time_limit():
    result = run_sa(TRAP, time_limit=0.0)
    assert result.status == "none"
    assert result.selected == []
    assert result.timed_out


def test_zero_steps_keeps_the_initial_cover():
    result = run_sa(TRAP, max_steps=0)
    assert result.selected == [0, 1, 2]
    assert result.extra["steps"] == 0
    assert not result.timed_out


def test_one_subset_cover_and_empty_candidate_subsets():
    result = run_sa(make_instance("one", 3, [[], [0, 1, 2], [0]]))
    assert result.selected == [1]
    assert result.extra["steps"] == 0


def test_only_fixed_subsets_stops_at_once():
    inst = make_instance("mandatory", 2, [[0], [1]])
    result = run_sa(inst)
    assert result.selected == [0, 1]
    assert result.extra["steps"] == 0
    assert not result.timed_out


def test_fixed_seed_and_step_limit_are_reproducible():
    inst = generate(30, 60, 0.15, seed=3)
    first = run_sa(inst, seed=7, max_steps=100)
    second = run_sa(inst, seed=7, max_steps=100)
    assert first.selected == second.selected
    assert first.extra["steps"] == second.extra["steps"] == 100
    assert first.extra["accepted_moves"] == second.extra["accepted_moves"]


def test_generated_covers_are_valid_and_never_worse_than_initialization():
    for seed in range(10):
        inst = generate(8, 10, 0.25, seed)
        before = copy.deepcopy(inst)
        optimum = brute_force.solve(inst, 10.0, np.random.default_rng(seed))
        result = run_sa(inst, seed=seed, max_steps=500)
        assert optimum.status == "optimal"
        assert optimum.size <= result.size <= result.extra["initial_size"]
        assert inst == before


@pytest.mark.parametrize("path", ["e/scpe1.txt", "clr/scpclr10.txt", "cyc/scpcyc06.txt", "sts/data.27"])
def test_benchmarks_through_the_shared_runner(path):
    record = run_one("simulated_annealing", BENCHMARKS / path, seed=0, time_limit=10.0, max_steps=500)
    assert record["valid"] and record["error"] is None
    assert record["status"] == "feasible"
    assert record["size"] <= record["extra"]["initial_size"]


@pytest.mark.parametrize("path", ["cyc/scpcyc11.txt", "sts/data.2187"])
def test_time_limit_on_large_instances(path):
    record = run_one("simulated_annealing", BENCHMARKS / path, seed=0, time_limit=1.0)
    assert record["valid"]
    assert record["timed_out"]
    assert record["time"] < 1.3


@pytest.mark.parametrize("params", [
    {"time_limit": -1.0}, {"time_limit": math.nan}, {"time_limit": math.inf},
    {"initial_temperature": 0.0}, {"initial_temperature": math.inf},
    {"cooling_rate": 0.0}, {"cooling_rate": 1.0}, {"cooling_rate": math.nan},
    {"min_acceptance": -0.1}, {"min_acceptance": 1.5}, {"min_acceptance": math.nan},
    {"frozen_levels": 0}, {"frozen_levels": 2.5},
    {"steps_per_temperature": 0}, {"steps_per_temperature": 1.5},
    {"max_steps": -1}, {"max_steps": 1.5},
])
def test_invalid_parameters_are_rejected(params):
    with pytest.raises(ValueError):
        run_sa(TRAP, **params)


def test_registry():
    assert METHODS["simulated_annealing"] is sa.solve