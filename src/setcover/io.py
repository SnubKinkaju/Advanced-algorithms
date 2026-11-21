"""Readers and writer for the benchmark files.

Both formats number the subsets from 1 to m. Every number is checked: a 0 or a
negative number would otherwise become a valid Python index counted from the
end of the list, and silently give another instance. A file with missing or
extra data is rejected too, with a message giving the file and the position.
"""

from pathlib import Path

from setcover.instance import Instance, make_instance


def _int(token: str, path: Path, what: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise ValueError(f"{path.name}: {what} is {token!r}, not an integer") from None


def _set_indices(tokens: list[str], m: int, path: Path, element: int) -> list[int]:
    """Turn the subset numbers of the file (1..m) into list indices (0..m-1)."""
    try:
        numbers = [int(t) for t in tokens]
    except ValueError:
        bad = next(t for t in tokens if not t.lstrip("-").isdigit())
        raise ValueError(f"{path.name}: element {element} lists {bad!r}, not an integer") from None
    if numbers and (min(numbers) < 1 or max(numbers) > m):
        j = next(j for j in numbers if not 1 <= j <= m)
        raise ValueError(f"{path.name}: element {element} refers to subset {j}, "
                         f"expected 1..{m}")
    return [j - 1 for j in numbers]


def _header(n: int, m: int, path: Path) -> None:
    if n < 0 or m < 0:
        raise ValueError(f"{path.name}: negative size in the header ({n} elements, {m} subsets)")


def read_orlib(path: str | Path) -> Instance:
    path = Path(path)
    tokens = path.read_text().split()
    if len(tokens) < 2:
        raise ValueError(f"{path.name}: the header must give the number of elements and of subsets")
    n_elements = _int(tokens[0], path, "the number of elements")
    n_sets = _int(tokens[1], path, "the number of subsets")
    _header(n_elements, n_sets, path)
    pos = 2 + n_sets
    if pos > len(tokens):
        raise ValueError(f"{path.name}: file ends early, in the costs of the subsets")
    if any(_int(c, path, "a cost") != 1 for c in tokens[2:pos]):
        raise ValueError(f"{path.name} is a weighted instance")
    sets = [[] for _ in range(n_sets)]
    for i in range(n_elements):
        if pos >= len(tokens):
            raise ValueError(f"{path.name}: file ends early, the count of element {i} is missing")
        try:
            k = int(tokens[pos])
        except ValueError:
            raise ValueError(f"{path.name}: the count of element {i} is {tokens[pos]!r}, "
                             f"not an integer") from None
        if k < 0:
            raise ValueError(f"{path.name}: element {i} has a negative count {k}")
        block = tokens[pos + 1:pos + 1 + k]
        if len(block) < k:
            raise ValueError(f"{path.name}: file ends early, element {i} announces {k} "
                             f"subsets but only {len(block)} follow")
        for j in _set_indices(block, n_sets, path, i):
            sets[j].append(i)
        pos += 1 + k
    if pos < len(tokens):
        raise ValueError(f"{path.name}: {len(tokens) - pos} extra values after the last element")
    return _checked(make_instance(path.stem, n_elements, sets))


def read_sts(path: str | Path) -> Instance:
    path = Path(path)
    lines = path.read_text().split("\n")
    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(f"{path.name}: the first line must give the number of subsets "
                         f"and the number of elements")
    n_sets = _int(header[0], path, "the number of subsets")
    n_elements = _int(header[1], path, "the number of elements")
    _header(n_elements, n_sets, path)
    if len(lines) - 1 < n_elements:
        raise ValueError(f"{path.name}: {n_elements} elements announced, "
                         f"only {len(lines) - 1} lines follow")
    sets = [[] for _ in range(n_sets)]
    for i in range(n_elements):
        for j in _set_indices(lines[i + 1].split(), n_sets, path, i):
            sets[j].append(i)
    if any(line.strip() for line in lines[n_elements + 1:]):
        raise ValueError(f"{path.name}: extra lines after the last element")
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