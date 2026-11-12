
import numpy as np

from setcover.instance import make_instance
from setcover.methods.dp_bottom_up import solve
from setcover.registry import METHODS
from setcover.result import verify


def run_dp(n, sets, time_limit=10.0):
    inst = make_instance("test", n, sets)
    rng = np.random.default_rng(0)
    result = solve(inst, time_limit, rng)
    verify(inst, result)
    return result


# TEST 1 : Solution optimale de taille 2
def test_simple():
    result = run_dp(
        4,
        [[0, 1], [1, 2], [2, 3]]
    )

    assert result.status == "optimal"
    assert result.size == 2
    assert set(result.selected) == {0, 2}


# TEST 2 : Un seul ensemble suffit
def test_one_set():
    result = run_dp(
        4,
        [[0, 1], [1, 2], [0, 1, 2, 3]]
    )

    assert result.status == "optimal"
    assert result.size == 1
    assert result.selected == [2]


# TEST 3 : Tous les ensembles sont nécessaires
def test_all_sets_needed():
    result = run_dp(
        3,
        [[0], [1], [2]]
    )

    assert result.status == "optimal"
    assert result.size == 3
    assert set(result.selected) == {0, 1, 2}


# TEST 4 : Couverture impossible
def test_impossible():
    result = run_dp(
        3,
        [[0], [1]]
    )

    assert result.status == "none"
    assert result.selected == []
    assert result.timed_out is False


# TEST 5 : Univers vide
def test_empty_universe():
    result = run_dp(
        0,
        []
    )

    assert result.status == "optimal"
    assert result.size == 0
    assert result.selected == []


# TEST 6 : Limite de temps nulle
def test_timeout():
    result = run_dp(
        4,
        [[0, 1], [1, 2], [2, 3]],
        time_limit=0.0
    )

    assert result.status == "none"
    assert result.timed_out is True


# TEST 7 : Enregistrement dans le registre
def test_registry():
    assert "dp_bottom_up" in METHODS
    assert METHODS["dp_bottom_up"] is solve
