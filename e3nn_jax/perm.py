from typing import Tuple, Set

import random
import math

TY_PERM = Tuple[int, ...]


def is_perm(p: TY_PERM):
    return sorted(set(p)) == list(range(len(p)))


def identity(n: int) -> TY_PERM:
    pass


def compose(p1: TY_PERM, p2: TY_PERM) -> TY_PERM:
    assert is_perm(p1) and is_perm(p2)
    assert len(p1) == len(p2)

    return tuple(p1[p2[i]] for i in range(len(p1)))


def inverse(p: TY_PERM) -> TY_PERM:
    return tuple(p.index(i) for i in range(len(p)))


def rand(n: int) -> TY_PERM:
    pass


def from_int(i: int, n: int) -> TY_PERM:
    pass


def to_int(p: TY_PERM) -> int:
    pass


def group(n: int) -> Set[TY_PERM]:
    pass


def germinate(subset: Set[TY_PERM]) -> Set[TY_PERM]:
    pass


def is_group(g: Set[TY_PERM]) -> bool:
    pass


def to_cycles(p: TY_PERM) -> Set[Tuple[int]]:
    n = len(p)

    cycles = set()

    for i in range(n):
        c = [i]
        while p[i] != c[0]:
            i = p[i]
            c += [i]
        if len(c) >= 2:
            i = c.index(min(c))
            c = c[i:] + c[:i]
            cycles.add(tuple(c))

    return cycles


def sign(p: TY_PERM) -> int:
    s = 1
    for c in to_cycles(p):
        if len(c) % 2 == 0:
            s = -s
    return s
