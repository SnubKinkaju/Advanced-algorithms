from pathlib import Path

import numpy as np

from setcover.io import read_instance
from setcover.registry import METHODS
from setcover.result import verify

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def test_trivial_gives_valid_cover_on_all_benchmarks():
    for f in BENCHMARKS.glob("*/*"):
        if f.name != "SOURCE.md":
            inst = read_instance(f)
            res = METHODS["trivial"](inst, 10.0, np.random.default_rng(0))
            verify(inst, res)
