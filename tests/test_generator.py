import pytest

from setcover.generator import generate
from setcover.io import read_instance, write_orlib


def test_same_seed_gives_same_instance():
    assert generate(30, 50, 0.1, seed=7) == generate(30, 50, 0.1, seed=7)


def test_different_seeds_give_different_instances():
    assert generate(30, 50, 0.1, seed=7).sets != generate(30, 50, 0.1, seed=8).sets


def test_every_element_covered_and_no_empty_set():
    inst = generate(40, 10, 0.01, seed=0)
    assert all(len(c) > 0 for c in inst.covering)
    assert all(len(s) > 0 for s in inst.sets)


def test_density_close_to_p():
    inst = generate(200, 300, 0.1, seed=1)
    assert abs(inst.density - 0.1) < 0.01


@pytest.mark.parametrize("n, m, p", [(0, 10, 0.1), (10, 0, 0.1), (10, 10, 0.0), (10, 10, 1.5)])
def test_invalid_parameters_are_refused(n, m, p):
    with pytest.raises(ValueError):
        generate(n, m, p, seed=0)


def test_write_then_read_gives_same_instance(tmp_path):
    inst = generate(25, 40, 0.15, seed=3)
    path = tmp_path / f"{inst.name}.txt"
    write_orlib(inst, path)
    assert read_instance(path) == inst
