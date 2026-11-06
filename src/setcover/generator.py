import numpy as np

from setcover.instance import Instance, make_instance


def generate(n: int, m: int, p: float, seed: int) -> Instance:
    if n < 1 or m < 1:
        raise ValueError("n and m must be at least 1")
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    rng = np.random.default_rng(seed)
    sets = [np.flatnonzero(rng.random(n) < p).tolist() for _ in range(m)]
    covered = [False] * n
    for s in sets:
        for e in s:
            covered[e] = True
    for e in range(n):
        if not covered[e]:
            sets[int(rng.integers(m))].append(e)
    for s in sets:
        if not s:
            s.append(int(rng.integers(n)))
    return make_instance(f"rand_n{n}_m{m}_p{p:g}_s{seed}", n, sets)
