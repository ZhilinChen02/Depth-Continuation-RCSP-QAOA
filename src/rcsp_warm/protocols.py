"""WARM / FROZEN / RANDOM initialisation and the budgeted public-energy fit.

Ported from analysis/rcsp_trainability_scaling_v2/train.py (study code r02, used unchanged through p=120).
  p=1 and RANDOM : x0 ~ Uniform(-4, 4)^(2p), seed = seed_for('random', task_id, mixer, seed, p)
  WARM   (p>1)   : x0 = [parent winner, tail], tail ~ Normal(0, 0.05)^2, seed = seed_for('warm-tail', task_id, mixer, seed, p);
                   the zero-layer anchor [parent, 0, 0] is evaluated first (energy only); all 2p angles are trained.
  FROZEN (p>1)   : prefix = parent winner (fixed); only the new pair is trained, from the same tail; anchor [0, 0].
Budget per fit: max(64, 4p) credits; energy-only call = 1 credit, energy + adjoint gradient = 3 credits.
Winner = lowest visited energy; an incumbent is replaced only on improvement > 1e-12 (earliest wins ties).
Box [-4, 4] for every angle (L-BFGS-B bounds; evaluated points are clipped); no angle wrapping.
This module never reads optimal-route labels.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.optimize import minimize

BOX = (-4.0, 4.0)
TIE = 1e-12
LBFGSB = dict(ftol=1e-12, gtol=1e-7, maxls=20, maxcor=10)


def seed_for(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2 ** 32)


def budget(p: int) -> int:
    return max(64, 4 * p)


def initial(task_id: str, mixer: str, seed: int, p: int, parent, strategy: str):
    """Return (active x0, anchor or None, fixed prefix or None)."""
    if p == 1 or strategy == "RANDOM_JOINT":
        return np.random.default_rng(seed_for("random", task_id, mixer, seed, p)).uniform(-4, 4, 2 * p), None, None
    parent = np.asarray(parent, float)
    tail = np.random.default_rng(seed_for("warm-tail", task_id, mixer, seed, p)).normal(0, 0.05, 2)
    if strategy == "FROZEN_LAYERWISE":
        return tail, np.zeros(2), parent
    if strategy == "WARM_JOINT":
        return np.r_[parent, tail], np.r_[parent, 0.0, 0.0], None
    raise ValueError(strategy)


class _Stop(Exception):
    pass


def fit(sim, x0, anchor=None, prefix=None, credit_budget=None):
    """Budgeted L-BFGS-B on the public energy. Returns a record dict (winner = lowest visited energy).
    Omits the original wall-clock guard and journal/resume machinery; accounting and search rules are identical."""
    a0 = np.asarray(x0, float)
    expand = (lambda z: np.asarray(z, float)) if prefix is None else (lambda z: np.r_[prefix, np.asarray(z, float)])
    p = len(expand(a0)) // 2
    B = budget(p) if credit_budget is None else credit_budget
    st = dict(calls=0, grads=0, credits=0, best=None, stop="BUDGET_EXHAUSTED", iterations=0)

    def evaluate(z, jac=True):
        cost = 3 if jac else 1
        if st["credits"] + cost > B:
            st["stop"] = "BUDGET_EXHAUSTED"; raise _Stop()
        a = expand(np.clip(np.asarray(z, float), *BOX))
        st["calls"] += 1; st["grads"] += int(jac); st["credits"] += cost
        if jac:
            f, g = sim.value_gradient(a)
            if prefix is not None:
                g = g[-2:]
        else:
            f, g = sim.value(a), None
        if not np.isfinite(f) or (g is not None and not np.all(np.isfinite(g))):
            raise FloatingPointError("non-finite loss/gradient")
        if st["best"] is None or f < st["best"]["loss"] - TIE:
            st["best"] = dict(loss=float(f), angles=a.tolist(), winner_call=st["calls"])
        return (float(f), g) if jac else float(f)

    result = None
    try:
        if anchor is not None:
            evaluate(anchor, False)
        if B - st["credits"] < 3:
            if B > st["credits"]:
                evaluate(a0, False)
        else:
            result = minimize(evaluate, a0, method="L-BFGS-B", jac=True, bounds=[BOX] * len(a0),
                              callback=lambda *a: st.__setitem__("iterations", st["iterations"] + 1),
                              options=dict(maxiter=B, maxfun=B, **LBFGSB))
            st["stop"] = "CONVERGED" if result.success else "SOLVER_STOP"
    except _Stop:
        pass
    best = dict(st["best"])
    best.update(p=p, status=st["stop"], objective_evaluations=st["calls"], gradient_evaluations=st["grads"],
                work_credits=st["credits"], nominal_layer_work=p * st["credits"], work_budget=B,
                active_parameter_count=len(a0), iterations=st["iterations"])
    return best
