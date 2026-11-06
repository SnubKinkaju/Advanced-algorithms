from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Instance:
    name: str
    n: int
    m: int
    sets: tuple[tuple[int, ...], ...]
    covering: tuple[tuple[int, ...], ...]

    @cached_property
    def masks(self) -> tuple[int, ...]:
        return tuple(sum(1 << e for e in s) for s in self.sets)

    @property
    def density(self) -> float:
        return sum(len(s) for s in self.sets) / (self.n * self.m)


def make_instance(name: str, n: int, sets: list[list[int]]) -> Instance:
    clean_sets = [tuple(sorted(set(s))) for s in sets]
    covering = [[] for _ in range(n)]
    for j, s in enumerate(clean_sets):
        for e in s:
            if e < 0 or e >= n:
                raise ValueError(f"element {e} of set {j} is outside 0..{n - 1}")
            covering[e].append(j)
    return Instance(
        name=name,
        n=n,
        m=len(clean_sets),
        sets=tuple(clean_sets),
        covering=tuple(tuple(c) for c in covering),
    )
