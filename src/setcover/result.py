from dataclasses import dataclass, field

from setcover.instance import Instance

STATUSES = ("optimal", "feasible", "none")


@dataclass
class Result:
    selected: list[int]
    size: int
    status: str
    timed_out: bool = False
    lower_bound: int | None = None
    nodes: int | None = None
    extra: dict = field(default_factory=dict)


def verify(inst: Instance, res: Result) -> None:
    if res.status not in STATUSES:
        raise ValueError(f"unknown status {res.status!r}")
    if res.status == "none":
        if res.selected:
            raise ValueError("status 'none' but subsets are selected")
        return
    if any(j < 0 or j >= inst.m for j in res.selected):
        raise ValueError("a selected index is outside 0..m-1")
    if len(set(res.selected)) != len(res.selected):
        raise ValueError("a subset is selected twice")
    if res.size != len(res.selected):
        raise ValueError(f"size is {res.size} but {len(res.selected)} subsets are selected")
    covered = [False] * inst.n
    for j in res.selected:
        for e in inst.sets[j]:
            covered[e] = True
    missing = covered.count(False)
    if missing:
        raise ValueError(f"{missing} element(s) not covered")
    if res.lower_bound is not None and res.lower_bound > res.size:
        raise ValueError("lower bound is greater than the size of the cover")
