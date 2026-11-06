from pathlib import Path

import numpy as np

from setcover.generator import generate
from setcover.instance import make_instance
from setcover.registry import METHODS
from setcover.result import verify
from setcover.runner import run_one

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def test_toy_instance_is_solved_optimally():
    inst = make_instance("toy", 4, [[0, 1], [1, 2], [2, 3]])
    res = METHODS["highs"](inst, 10.0, np.random.default_rng(0))
    verify(inst, res)
    assert res.status == "optimal"
    assert res.size == 2


def test_sts27_optimum_is_18():
    record = run_one("highs", BENCHMARKS / "sts" / "data.27", seed=0, time_limit=60.0)
    assert record["valid"]
    assert record["status"] == "optimal"
    assert record["size"] == 18


def test_trivial_is_never_below_highs_optimum():
    for seed in range(10):
        inst = generate(12, 20, 0.2, seed=seed)
        best = METHODS["highs"](inst, 10.0, np.random.default_rng(seed))
        other = METHODS["trivial"](inst, 10.0, np.random.default_rng(seed))
        assert best.status == "optimal"
        assert other.size >= best.size


def test_time_limit_is_respected_on_a_hard_instance():
    record = run_one("highs", BENCHMARKS / "clr" / "scpclr10.txt", seed=0, time_limit=1.0)
    assert record["valid"]
    assert record["timed_out"]
    assert record["status"] in ("feasible", "none")
    if record["status"] == "feasible":
        assert record["lower_bound"] <= record["size"]
    assert record["time"] < 5.0
