
import numpy as np

from setcover.instance import make_instance
from setcover.result import verify
from setcover.registry import METHODS
from setcover.methods.dp_memoized import solve


def run_dp(n, sets, time_limit=5.0):
    inst = make_instance("test", n, sets)
    rng = np.random.default_rng(42)
    result = solve(inst, time_limit, rng)
    return inst, result


def test_simple():
    inst, result = run_dp(
        4, [[0, 1], [1, 2], [2, 3]]
    )
    assert result.status == "optimal"
    assert result.size == 2
    assert set(result.selected) == {0, 2}
    verify(inst, result)


def test_one_set():
    inst, result = run_dp(
        4, [[0, 1], [1, 2], [0, 1, 2, 3]]
    )
    assert result.status == "optimal"
    assert result.size == 1
    assert result.selected == [2]
    verify(inst, result)


def test_all_sets_needed():
    inst, result = run_dp(
        3, [[0], [1], [2]]
    )
    assert result.status == "optimal"
    assert result.size == 3
    verify(inst, result)


def test_impossible():
    inst, result = run_dp(
        3, [[0], [1]]
    )
    assert result.status == "none"
    assert result.timed_out is False


def test_empty_universe():
    inst, result = run_dp(0, [])
    assert result.status == "optimal"
    assert result.size == 0
    assert result.selected == []
    verify(inst, result)


def test_timeout():
    inst, result = run_dp(
        4, [[0, 1], [1, 2], [2, 3]],
        time_limit=0.0
    )
    assert result.status == "none"
    assert result.timed_out is True


def test_registry():
    assert "dp_memoized" in METHODS
    assert METHODS["dp_memoized"] is solve
