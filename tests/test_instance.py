import dataclasses
from pathlib import Path

import pytest

from setcover.generator import generate
from setcover.instance import make_instance
from setcover.io import read_instance

BENCHMARKS = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
BENCHMARK_FILES = sorted(p for p in BENCHMARKS.glob("*/*") if p.name != "SOURCE.md")


def test_toy_instance():
    inst = make_instance("toy", 4, [[0, 1], [1, 2, 2], [2, 3]])
    assert inst.m == 3
    assert inst.sets == ((0, 1), (1, 2), (2, 3))
    assert inst.covering == ((0,), (0, 1), (1, 2), (2,))
    assert inst.masks == (0b11, 0b110, 0b1100)


def test_instance_is_frozen():
    inst = make_instance("toy", 4, [[0, 1], [1, 2], [2, 3]])
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.n = 5


def test_element_out_of_range():
    with pytest.raises(ValueError):
        make_instance("bad", 3, [[0, 3]])


def test_masks_match_the_definition_on_generated_instances():
    for seed in range(20):
        inst = generate(n=40, m=30, p=0.2, seed=seed)
        assert inst.masks == tuple(sum(1 << e for e in s) for s in inst.sets)


def test_masks_with_elements_on_byte_boundaries():
    inst = make_instance("edges", 17, [[0], [7], [8], [15], [16], [0, 7, 8, 15, 16]])
    assert inst.masks == (1, 1 << 7, 1 << 8, 1 << 15, 1 << 16, 0b11000000110000001)


def test_benchmark_list_is_complete():
    assert len(BENCHMARK_FILES) == 26


@pytest.mark.parametrize("path", BENCHMARK_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_masks_match_the_sets_on_benchmarks(path):
    # Read the bits back from the binary string, which costs O(n) per mask.
    inst = read_instance(path)
    assert len(inst.masks) == inst.m
    for s, mask in zip(inst.sets, inst.masks):
        bits = bin(mask)[2:][::-1]
        assert tuple(i for i, b in enumerate(bits) if b == "1") == s