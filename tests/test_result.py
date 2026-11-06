import pytest

from setcover.instance import make_instance
from setcover.result import Result, verify

TOY = make_instance("toy", 4, [[0, 1], [1, 2], [2, 3]])


def test_valid_cover_is_accepted():
    verify(TOY, Result(selected=[0, 2], size=2, status="optimal"))


@pytest.mark.parametrize(
    "result",
    [
        Result([0, 1], 2, "feasible"),
        Result([0, 0, 2], 3, "feasible"),
        Result([0, 2], 3, "feasible"),
        Result([0, 5], 2, "feasible"),
        Result([0, 2], 2, "best"),
        Result([0, 2], 2, "feasible", lower_bound=3),
    ],
)
def test_invalid_results_are_refused(result):
    with pytest.raises(ValueError):
        verify(TOY, result)
