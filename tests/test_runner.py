from pathlib import Path

import pytest

from setcover.runner import run_one

E1 = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "e" / "scpe1.txt"


def test_run_one_on_e1():
    record = run_one("trivial", E1, seed=0, time_limit=10.0)
    assert record["valid"]
    assert record["error"] is None
    assert record["size"] == 10
    assert record["status"] == "feasible"
    assert record["time"] >= 0


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        run_one("does_not_exist", E1, seed=0, time_limit=10.0)


def test_same_seed_gives_same_result():
    first = run_one("trivial", E1, seed=42, time_limit=10.0)
    second = run_one("trivial", E1, seed=42, time_limit=10.0)
    assert first["selected"] == second["selected"]


def test_mask_construction_is_not_timed():
    # Building the masks of sts/data.2187 takes about 0.5 s; with a limit of
    # 0.05 s the brute force used to be measured at about 0.6 s.
    path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "sts" / "data.2187"
    record = run_one("brute_force", path, seed=0, time_limit=0.05)
    assert record["valid"]
    assert record["timed_out"]
    assert record["time"] < 0.2
    assert record["setup_time"] > record["time"]