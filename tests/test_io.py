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


# Malformed files. Subsets are numbered from 1 to m in both formats.
ORLIB_GOOD = "3 3\n1 1 1\n1 1\n1 2\n1 3\n"
STS_GOOD = "3 3\n1 2\n1 3\n2 3\n"


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_valid_small_files(tmp_path):
    assert read_instance(write(tmp_path, "good.txt", ORLIB_GOOD)).sets == ((0,), (1,), (2,))
    assert read_instance(write(tmp_path, "data.3", STS_GOOD)).sets == ((0, 1), (0, 2), (1, 2))


@pytest.mark.parametrize("name, text, message", [
    # A 0 or a negative number used to become a Python index counted from the end.
    ("zero.txt", "3 3\n1 1 1\n1 1\n1 2\n1 0\n", "element 2 refers to subset 0, expected 1..3"),
    ("negative.txt", "3 3\n1 1 1\n1 1\n1 2\n1 -1\n", "element 2 refers to subset -1"),
    ("too_big.txt", "3 3\n1 1 1\n1 1\n1 2\n1 4\n", "element 2 refers to subset 4"),
    ("extra.txt", ORLIB_GOOD + "99 98\n", "2 extra values after the last element"),
    ("short.txt", "3 3\n1 1 1\n1 1\n1 2\n", "file ends early, the count of element 2"),
    ("wrong_count.txt", "3 3\n1 1 1\n2 1\n1 2\n1 3\n", "file ends early"),
    ("not_int.txt", "3 3\n1 1 1\n1 1\n1 x\n1 3\n", "'x', not an integer"),
    ("neg_header.txt", "-3 3\n1 1 1\n", "negative size"),
    ("short_costs.txt", "3 3\n1 1\n", "file ends early, in the costs"),
])
def test_malformed_orlib_files_are_rejected(tmp_path, name, text, message):
    with pytest.raises(ValueError, match=message):
        read_instance(write(tmp_path, name, text))


@pytest.mark.parametrize("name, text, message", [
    ("data.zero", "3 3\n1 2\n1 3\n0 2\n", "element 2 refers to subset 0"),
    ("data.big", "3 3\n1 2\n1 3\n2 4\n", "element 2 refers to subset 4"),
    ("data.extra", STS_GOOD + "1 2\n", "extra lines after the last element"),
    ("data.short", "3 3\n1 2\n1 3", "3 elements announced, only 2 lines follow"),
    ("data.header", "3\n1 2\n", "first line must give"),
])
def test_malformed_sts_files_are_rejected(tmp_path, name, text, message):
    with pytest.raises(ValueError, match=message):
        read_instance(write(tmp_path, name, text))


def test_trailing_blank_lines_are_accepted(tmp_path):
    assert read_instance(write(tmp_path, "data.3", STS_GOOD + "\n\n")).m == 3
    assert read_instance(write(tmp_path, "good.txt", ORLIB_GOOD + "\n\n")).m == 3