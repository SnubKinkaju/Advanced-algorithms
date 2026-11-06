from pathlib import Path

import pytest

from setcover.io import read_instance

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"

EXPECTED_SIZES = [
    ("e/scpe1.txt", 50, 500),
    ("clr/scpclr10.txt", 511, 210),
    ("cyc/scpcyc06.txt", 240, 192),
    ("cyc/scpcyc11.txt", 28160, 11264),
    ("sts/data.27", 117, 27),
    ("sts/data.81", 1080, 81),
]


@pytest.mark.parametrize("relative_path, n, m", EXPECTED_SIZES)
def test_known_sizes(relative_path, n, m):
    inst = read_instance(BENCHMARKS / relative_path)
    assert (inst.n, inst.m) == (n, m)


def test_all_benchmarks_load():
    files = [f for f in BENCHMARKS.glob("*/*") if f.name != "SOURCE.md"]
    assert len(files) == 26
    for f in files:
        read_instance(f)
