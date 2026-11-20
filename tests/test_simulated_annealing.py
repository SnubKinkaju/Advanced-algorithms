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


class FixedRandom:
    """Control proposed moves and acceptance draws in small examples."""

    def __init__(self, moves=(), draw=0.0):
        self.moves = iter(moves)
        self.draw = draw

    def integers(self, high):
        return next(self.moves)

    def random(self):
        return self.draw


def test_escapes_a_cover_that_cannot_be_improved_by_removals_alone():
    initial = greedy.solve(TRAP, 10.0, np.random.default_rng(0))
    assert initial.selected == [0, 1, 2]
    for removed in initial.selected:
        covered = {e for j in initial.selected if j != removed for e in TRAP.sets[j]}
        assert len(covered) < TRAP.n

    result = run_sa(TRAP)
    assert result.status == "feasible"
    assert result.selected == [3, 4]
    assert result.extra["initial_size"] == 3


@pytest.mark.parametrize("temperature, accepted", [(1.0, True), (0.1, False)])
def test_worse_move_acceptance_depends_on_temperature(temperature, accepted):
    inst = make_instance("duplicate", 1, [[0], [0]])
    selected, counts = {0}, [1]
    changed = sa._try_move(inst, selected, counts, 1, temperature, FixedRandom(draw=0.2))
    assert changed is accepted
    assert selected == ({0, 1} if accepted else {0})
    assert counts == ([2] if accepted else [1])


def test_removal_preserves_coverage_and_improving_moves_are_always_accepted():
    inst = make_instance("small", 2, [[0, 1], [1]])
    selected, counts = {0, 1}, [1, 2]
    assert not sa._try_move(inst, selected, counts, 0, 0.01, FixedRandom(draw=1.0))
    assert selected == {0, 1} and counts == [1, 2]
    assert sa._try_move(inst, selected, counts, 1, 0.01, FixedRandom(draw=1.0))
    assert selected == {0} and counts == [1, 1]


def test_counts_match_independent_recomputation_after_random_moves():
    inst = generate(20, 30, 0.2, seed=4)
    selected = set(range(inst.m))
    counts = sa._coverage_counts(inst, selected, time.perf_counter() + 10)
    rng = np.random.default_rng(3)
    for _ in range(300):
        sa._try_move(inst, selected, counts, int(rng.integers(inst.m)), 1.0, rng)
        expected = [sum(e in inst.sets[j] for j in selected) for e in range(inst.n)]
        assert counts == expected
        assert all(count > 0 for count in counts)


def test_returns_best_cover_even_when_the_last_move_is_worse():
    inst = make_instance("small", 2, [[0], [1], [0, 1]])
    result = sa._anneal(
        inst, [0, 1, 2], time.perf_counter() + 10, FixedRandom(moves=[0, 0]),
        1.0, 0.95, 0.01, 10, 2,
    )
    verify(inst, result)
    assert result.selected == [1, 2]
    assert result.extra["accepted_moves"] == 2


def test_improvement_is_saved_before_timeout(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(sa.time, "perf_counter", lambda: clock[0])
    original_move = sa._try_move

    def slow_move(*args):
        changed = original_move(*args)
        clock[0] = 2.0
        return changed

    monkeypatch.setattr(sa, "_try_move", slow_move)
    inst = make_instance("small", 2, [[0], [1], [0, 1]])
    result = sa._anneal(inst, [0, 1, 2], 1.0, FixedRandom(moves=[0]), 1.0, 0.95, 0.01, 10, None)
    verify(inst, result)
    assert result.selected == [1, 2]
    assert result.timed_out


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


def test_cooling_schedule_stops_without_claiming_a_timeout():
    inst = make_instance("mandatory", 2, [[0], [1]])
    result = run_sa(inst, initial_temperature=1.0, cooling_rate=0.5,
                    min_temperature=0.2, steps_per_temperature=4)
    assert result.extra["steps"] == 12
    assert result.extra["temperature"] == pytest.approx(0.125)
    assert not result.timed_out


def test_fixed_seed_and_step_limit_are_reproducible():
    inst = generate(30, 60, 0.15, seed=3)
    first = run_sa(inst, seed=7, max_steps=100)
    second = run_sa(inst, seed=7, max_steps=100)
    assert first.selected == second.selected
    assert first.extra == second.extra
    assert first.extra["steps"] == 100


def test_generated_covers_are_valid_and_never_worse_than_initialization():
    for seed in range(10):
        inst = generate(8, 10, 0.25, seed)
        before = copy.deepcopy(inst)
        optimum = brute_force.solve(inst, 10.0, np.random.default_rng(seed))
        result = run_sa(inst, seed=seed, max_steps=500)
        assert optimum.status == "optimal"
        assert result.status == "feasible"
        assert optimum.size <= result.size <= result.extra["initial_size"]
        assert inst == before


@pytest.mark.parametrize("path", ["e/scpe1.txt", "clr/scpclr10.txt", "cyc/scpcyc06.txt", "sts/data.27"])
def test_benchmarks_through_the_shared_runner(path):
    record = run_one("simulated_annealing", BENCHMARKS / path, seed=0, time_limit=10.0, max_steps=500)
    assert record["valid"] and record["error"] is None
    assert record["status"] == "feasible"
    assert record["size"] <= record["extra"]["initial_size"]


def test_short_time_limit_on_large_instance():
    record = run_one("simulated_annealing", BENCHMARKS / "cyc/scpcyc11.txt", seed=0, time_limit=0.05)
    assert record["valid"]
    assert record["timed_out"]
    assert record["time"] < 1.0


@pytest.mark.parametrize("params", [
    {"time_limit": -1.0}, {"time_limit": math.nan}, {"time_limit": math.inf},
    {"initial_temperature": 0.0}, {"initial_temperature": math.inf},
    {"cooling_rate": 0.0}, {"cooling_rate": 1.0}, {"cooling_rate": math.nan},
    {"min_temperature": 0.0}, {"min_temperature": 1.0},
    {"min_temperature": math.nan}, {"steps_per_temperature": 0},
    {"steps_per_temperature": 1.5}, {"max_steps": -1}, {"max_steps": 1.5},
])
def test_invalid_parameters_are_rejected(params):
    with pytest.raises(ValueError):
        run_sa(TRAP, **params)


def test_registry():
    assert METHODS["simulated_annealing"] is sa.solve
