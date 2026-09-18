"""Public RCSP instance data and the full-space cost Hamiltonian.

Bit convention: edge j is bit j of the basis index x (bit 0 least significant).
A state x is *feasible* iff its selected edges form exactly one simple directed
s-t path (no extra edges, no detached cycles) whose total resource is within budget.
The feasibility predicate is part of the public problem definition.

Cost Hamiltonian (as implemented in the original study, representation FOREST, penalty BOOLEAN):
  d      = forest-potential-shifted edge costs (costs on a deterministic spanning forest become 0;
           every s-t path is shifted by the same constant, so the optimal set is unchanged)
  L      = sum_j min(0, d_j),  W = sum_j |d_j|,  M = 1 + W
  h(x)   = ((d.x - L)/M + 2*[x infeasible]) / 3
  bound  = ((W/M + 2)/3) / 2
  H(x)   = (h(x) - bound) / bound      in [-1, 1];  training loss = <psi|H|psi>
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def load_task(path) -> dict:
    return json.loads(Path(path).read_text())


def forest_transform(task: dict) -> list[int]:
    """Port of global_opt_flow_v3_causal_audit/audit.py::forest_transform (topology + public costs only)."""
    n, edges, c = task["n"], task["edges"], [int(x) for x in task["costs"]]
    parent = list(range(n)); tree = [[] for _ in range(n)]; chosen = []

    def root(v):
        while parent[v] != v:
            v = parent[v]
        return v

    for j, (u, v) in enumerate(edges):
        ru, rv = root(u), root(v)
        if ru != rv:
            parent[ru] = rv; tree[u].append((v, -c[j])); tree[v].append((u, c[j])); chosen.append(j)
    pot = [None] * n
    for r in range(n):
        if pot[r] is not None:
            continue
        pot[r] = 0; stack = [r]
        while stack:
            u = stack.pop()
            for v, change in tree[u]:
                if pot[v] is None:
                    pot[v] = pot[u] + change; stack.append(v)
    new = [c[j] + pot[v] - pot[u] for j, (u, v) in enumerate(edges)]
    assert all(new[j] == 0 for j in chosen)
    return new


def public_feasibility(task: dict) -> np.ndarray:
    """Per-state route predicate (port of depth_joint_scaling_v2/kernel.py::public_arrays). O(2^m m)."""
    n, s, t = task["n"], task["s"], task["t"]; edges = task["edges"]; res = task["resources"]; m = len(edges)
    N = 1 << m; x = np.arange(N, dtype=np.int64)
    bits = ((x[:, None] >> np.arange(m)) & 1).astype(np.int64)
    r = bits @ np.asarray(res, dtype=np.int64); count = bits.sum(1)
    inc = np.zeros((N, n), np.int64); out = np.zeros((N, n), np.int64)
    for j, (u, v) in enumerate(edges):
        out[:, u] += bits[:, j]; inc[:, v] += bits[:, j]
    ok = (r <= task["budget"]) & (inc[:, s] == 0) & (out[:, s] == 1) & (inc[:, t] == 1) & (out[:, t] == 0)
    for u in range(n):
        if u in (s, t):
            continue
        ok &= ((inc[:, u] == 0) & (out[:, u] == 0)) | ((inc[:, u] == 1) & (out[:, u] == 1))
    feas = np.zeros(N, bool)
    for xi in np.flatnonzero(ok):          # walk the unique successor chain from s
        following = {}
        for j, (u, v) in enumerate(edges):
            if (xi >> j) & 1:
                following[u] = v
        u = s; steps = 0
        for _ in range(n):
            if u == t or u not in following:
                break
            u = following[u]; steps += 1
        feas[xi] = (u == t and steps == count[xi])
    return feas


@dataclass
class CostHamiltonian:
    H: np.ndarray            # normalised diagonal, training-visible
    feasible: np.ndarray     # public predicate
    forest_costs: list
    L: int
    W: int
    M: int


def cost_hamiltonian(task: dict) -> CostHamiltonian:
    d = forest_transform(task); m = len(d); N = 1 << m
    x = np.arange(N, dtype=np.int64)
    bits = ((x[:, None] >> np.arange(m)) & 1).astype(np.int64)
    cost = bits @ np.asarray(d, dtype=np.int64)
    f = public_feasibility(task)
    L = sum(min(0, c) for c in d); W = sum(abs(c) for c in d); M = 1 + W
    h = ((cost - L) / M + 2 * (~f)) / 3
    upper = (W / M + 2) / 3; bound = upper / 2
    return CostHamiltonian(np.ascontiguousarray((h - bound) / bound), f, d, L, W, M)
