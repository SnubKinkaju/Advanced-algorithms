from pathlib import Path

from setcover.instance import Instance, make_instance


def read_orlib(path: str | Path) -> Instance:
    path = Path(path)
    tokens = iter(path.read_text().split())
    n_elements = int(next(tokens))
    n_sets = int(next(tokens))
    costs = [int(next(tokens)) for _ in range(n_sets)]
    if any(c != 1 for c in costs):
        raise ValueError(f"{path.name} is a weighted instance")
    sets = [[] for _ in range(n_sets)]
    for i in range(n_elements):
        k = int(next(tokens))
        for _ in range(k):
            sets[int(next(tokens)) - 1].append(i)
    return _checked(make_instance(path.stem, n_elements, sets))


def read_sts(path: str | Path) -> Instance:
    path = Path(path)
    lines = path.read_text().split("\n")
    n_sets, n_elements = (int(x) for x in lines[0].split())
    sets = [[] for _ in range(n_sets)]
    for i in range(n_elements):
        for j in lines[i + 1].split():
            sets[int(j) - 1].append(i)
    return _checked(make_instance("sts" + path.suffix[1:], n_elements, sets))


def read_instance(path: str | Path) -> Instance:
    path = Path(path)
    if path.name.startswith("data."):
        return read_sts(path)
    return read_orlib(path)


def _checked(inst: Instance) -> Instance:
    uncovered = [i for i, c in enumerate(inst.covering) if len(c) == 0]
    if uncovered:
        raise ValueError(f"{inst.name}: elements {uncovered[:5]} are covered by no set")
    return inst


def write_orlib(inst: Instance, path: str | Path) -> None:
    lines = [f"{inst.n} {inst.m}", " ".join(["1"] * inst.m)]
    for c in inst.covering:
        lines.append(" ".join([str(len(c))] + [str(j + 1) for j in c]))
    Path(path).write_text("\n".join(lines) + "\n")
