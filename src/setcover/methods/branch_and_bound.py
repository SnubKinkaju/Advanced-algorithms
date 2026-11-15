import math
import time

import numpy as np

from setcover.bounds import lp_bound
from setcover.instance import Instance
from setcover.result import Result


def _bits(x: int):
    """Indices des bits à 1 de x, donc les éléments d'un masque."""
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def _greedy(masks: tuple[int, ...], full: int) -> list[int]:
    """Solution initiale : prend toujours l'ensemble qui couvre le plus d'éléments restants."""
    uncovered = full
    chosen = []
    while uncovered:
        j = max(range(len(masks)), key=lambda k: (masks[k] & uncovered).bit_count())
        chosen.append(j)
        uncovered &= ~masks[j]
    return chosen


def _unwind(chosen) -> list[int]:
    """Reconstruit la liste des ensembles choisis à partir de la liste chaînée."""
    out = []
    while chosen is not None:
        j, chosen = chosen
        out.append(j)
    return out


def solve(inst: Instance, time_limit: float, rng: np.random.Generator,
          use_lp: bool = True) -> Result:
    deadline = time.perf_counter() + time_limit

    # Un élément qui n'apparaît dans aucun ensemble rend le problème impossible.
    if any(not c for c in inst.covering):
        return Result(selected=[], size=0, status="none")
    if inst.n == 0:
        return Result(selected=[], size=0, status="optimal", lower_bound=0, nodes=0)

    masks = inst.masks
    full = (1 << inst.n) - 1

    # Borne supérieure initiale (glouton) et borne inférieure à la racine (LP).
    best = _greedy(masks, full)
    root_lb = lp_bound(inst) if use_lp else 1
    if len(best) <= root_lb:
        return Result(selected=sorted(best), size=len(best), status="optimal",
                      lower_bound=len(best), nodes=0)

    # Parcours en profondeur avec une pile explicite.
    # Un nœud = (éléments non couverts, ensembles choisis, profondeur, ensembles interdits).
    stack = [(full, None, 0, 0)]
    nodes = 0
    timed_out = False

    while stack:
        if time.perf_counter() >= deadline:
            timed_out = True
            break

        uncovered, chosen, depth, forbidden = stack.pop()
        nodes += 1

        # Feuille : tout est couvert.
        if uncovered == 0:
            if depth < len(best):
                best = _unwind(chosen)
            continue

        # Il faut au moins un ensemble de plus : inutile si on ne peut plus faire mieux.
        if depth + 1 >= len(best):
            continue

        # Choix de l'élément à couvrir : celui qui a le moins d'ensembles autorisés.
        branch = None
        for e in _bits(uncovered):
            cands = [j for j in inst.covering[e] if not (forbidden >> j) & 1]
            if not cands:          # élément impossible à couvrir dans cette branche
                branch = []
                break
            if branch is None or len(cands) < len(branch):
                branch = cands
                if len(cands) == 1:
                    break
        if not branch:
            continue

        # Borne inférieure : il reste u éléments, un ensemble en couvre au plus k.
        max_gain = max((masks[j] & uncovered).bit_count()
                       for j in range(inst.m) if not (forbidden >> j) & 1)
        if depth + math.ceil(uncovered.bit_count() / max_gain) >= len(best):
            continue

        # Les ensembles les plus prometteurs d'abord, comme dans le cours.
        branch.sort(key=lambda j: (masks[j] & uncovered).bit_count(), reverse=True)

        # Branche i : on prend branch[i] et on interdit branch[0..i-1],
        # déjà explorés, pour ne jamais visiter deux fois la même solution.
        children = []
        f = forbidden
        for j in branch:
            children.append((uncovered & ~masks[j], (j, chosen), depth + 1, f))
            f |= 1 << j
        stack.extend(reversed(children))

    if timed_out:
        return Result(selected=sorted(best), size=len(best), status="feasible",
                      timed_out=True, lower_bound=root_lb, nodes=nodes)
    return Result(selected=sorted(best), size=len(best), status="optimal",
                  lower_bound=len(best), nodes=nodes)