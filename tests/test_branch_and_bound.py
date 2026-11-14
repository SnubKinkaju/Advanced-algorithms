import numpy as np

from setcover.instance import make_instance
from setcover.methods import branch_and_bound, highs_ref
from setcover.result import verify


def run(inst, time_limit=10.0):
    return branch_and_bound.solve(inst, time_limit, np.random.default_rng(0))


def test_small_instance():
    inst = make_instance("toy", 5, [[0, 1, 2], [2, 3], [3, 4], [0, 4], [1]])
    res = run(inst)
    verify(inst, res)
    assert res.status == "optimal"
    assert res.size == 2


def test_beats_greedy():
    # Le glouton prend {0,1,2,3} puis deux autres ensembles : 3 au total.
    # L'optimum est {0,1,4} + {2,3,5} : 2 ensembles.
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


def test_timeout_returns_valid_cover():
    inst = make_instance("trap", 6, [[0, 1, 2, 3], [0, 1, 4], [2, 3, 5]])
    res = run(inst, time_limit=0.0)
    verify(inst, res)
    assert res.timed_out
    assert res.status == "feasible"


def test_matches_highs_on_random_instances():
    rng = np.random.default_rng(42)
    for k in range(20):
        n, m = 12, 15
        sets = [[e for e in range(n) if rng.random() < 0.3] for _ in range(m)]
        for e in range(n):                      # chaque élément est couvert au moins une fois
            sets[rng.integers(m)].append(e)
        inst = make_instance(f"rand{k}", n, sets)
        res = run(inst)
        ref = highs_ref.solve(inst, 10.0, np.random.default_rng(0))
        verify(inst, res)
        assert res.status == "optimal"
        assert res.size == ref.size