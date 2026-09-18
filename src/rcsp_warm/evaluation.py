"""Evaluation side: optimal routes and probability metrics. Not imported by training code.

P_opt counts every optimal feasible bit string (all ties). P_opt = P_feas * P(opt | feas) holds per state;
P(opt | feas) is undefined (None) when P_feas = 0.
"""
from __future__ import annotations
import numpy as np


def enumerate_routes(task: dict) -> list[dict]:
    """Independent DFS over simple directed s-t paths (does not use problem.public_feasibility)."""
    adj = [[] for _ in range(task["n"])]
    for j, (u, v) in enumerate(task["edges"]):
        adj[u].append((v, j))
    routes = []

    def visit(u, seen, mask):
        if u == task["t"]:
            routes.append(mask); return
        for v, j in adj[u]:
            if v not in seen:
                visit(v, seen | {v}, mask | (1 << j))
    visit(task["s"], {task["s"]}, 0)
    out = []
    for mask in routes:
        sel = [j for j in range(len(task["edges"])) if (mask >> j) & 1]
        out.append(dict(mask=mask, cost=sum(task["costs"][j] for j in sel), resource=sum(task["resources"][j] for j in sel)))
    return out


def labels(task: dict) -> dict:
    routes = enumerate_routes(task)
    feas = [r for r in routes if r["resource"] <= task["budget"]]
    c = min(r["cost"] for r in feas)
    return dict(routes=routes, feasible_masks=[r["mask"] for r in feas], optimal_masks=[r["mask"] for r in feas if r["cost"] == c],
                optimal_cost=c, route_count=len(routes), feasible_count=len(feas))


def metrics(psi: np.ndarray, H: np.ndarray, lab: dict) -> dict:
    pr = np.abs(psi) ** 2
    pf = float(pr[lab["feasible_masks"]].sum()); po = float(pr[lab["optimal_masks"]].sum())
    ranked = sorted(lab["feasible_masks"], key=lambda s: -pr[s])
    return dict(p_opt=po, p_feas=pf, p_opt_given_feas=po / pf if pf > 0 else None, energy=float(pr @ H),
                norm_error=float(abs(pr.sum() - 1)), mass_optimal=po, mass_feasible_nonoptimal=pf - po, mass_infeasible=1 - pf,
                top_feasible=[(int(s), float(pr[s])) for s in ranked[:3]])
