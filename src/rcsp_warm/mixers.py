"""X and graph multi-X mixers.

X      : H_M = -(1/m) sum_j X_j
MULTI  : H_M = -(1/2m) sum_j X_j - (1/2K) sum_{s in S} X^s,   X^s = prod_{j in s} X_j,  K = |S|
S = graph_masks(task): topology-only masks (maximal degree-two corridors, fundamental cycles of a
deterministic spanning forest, incident edge pairs), at most 2m, size/rule order.
Both mixers are diagonal in the Hadamard basis; `spectrum` returns the eigenvalue of every Walsh index x:
  eigen[x] = sum_s coef_s (-1)^{popcount(x & s)}.
"""
from __future__ import annotations
import itertools
import numpy as np


def graph_masks(task: dict, cap_factor: int = 2) -> list[int]:
    """Port of global_opt_flow_v2/core.py::graph_masks."""
    edges, n = task["edges"], task["n"]; m = len(edges)
    inc, outgoing, incoming = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for j, (u, v) in enumerate(edges):
        inc[u].append(j); inc[v].append(j); outgoing[u].append(j); incoming[v].append(j)
    corridors = set()
    for j, (u, v) in enumerate(edges):
        chain, seen, current = [j], {j}, v
        while len(incoming[current]) == len(outgoing[current]) == 1:
            nxt = outgoing[current][0]
            if nxt in seen:
                break
            chain.append(nxt); seen.add(nxt); current = edges[nxt][1]
        if len(chain) > 1:
            corridors.add(sum(1 << i for i in chain))
    tree = [[] for _ in range(n)]; parent = list(range(n))

    def root(u):
        while parent[u] != u:
            u = parent[u]
        return u
    cycles = set()
    for j, (u, v) in enumerate(edges):
        a, b = root(u), root(v)
        if a != b:
            parent[a] = b; tree[u].append((v, j)); tree[v].append((u, j))
        else:
            stack = [(u, -1, 0)]
            while stack:
                cur, prev, mask = stack.pop()
                if cur == v:
                    cycles.add(mask | (1 << j)); break
                stack.extend((nxt, cur, mask | (1 << k)) for nxt, k in tree[cur] if nxt != prev)
    pairs = {(1 << i) | (1 << j) for group in inc for i, j in itertools.combinations(group, 2)}
    maximal = {s for s in corridors if not any(s != t and s & t == s for t in corridors)}
    ordered = sorted(maximal, key=lambda s: (-s.bit_count(), s)) + sorted(cycles, key=lambda s: (s.bit_count(), s)) + sorted(pairs)
    return list(dict.fromkeys(s for s in ordered if s.bit_count() > 1))[: cap_factor * m]


def mixer_terms(task: dict, mixer: str) -> list[tuple[int, float]]:
    m = len(task["edges"])
    if mixer == "X":
        return [(1 << j, -1.0 / m) for j in range(m)]
    if mixer == "MULTI":
        masks = graph_masks(task)
        return [(1 << j, -0.5 / m) for j in range(m)] + [(s, -0.5 / len(masks)) for s in masks]
    raise ValueError(f"unknown mixer {mixer!r}")


def spectrum(task: dict, mixer: str) -> np.ndarray:
    m = len(task["edges"]); x = np.arange(1 << m, dtype=np.int64); eig = np.zeros(1 << m)
    coef: dict[int, float] = {}
    for s, w in mixer_terms(task, mixer):
        coef[s] = coef.get(s, 0.0) + w
    for s, w in coef.items():
        parity = np.zeros(1 << m, dtype=np.int64); y = x & s
        for j in range(m):
            parity ^= (y >> j) & 1
        eig += w * (1 - 2 * parity)
    return eig
