import dataclasses

import pytest

from setcover.instance import make_instance


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
