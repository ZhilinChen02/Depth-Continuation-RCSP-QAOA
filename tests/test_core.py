import itertools, json, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm
from rcsp_warm import records
from rcsp_warm.problem import cost_hamiltonian, public_feasibility, forest_transform
from rcsp_warm.mixers import spectrum, mixer_terms
from rcsp_warm.simulator import QAOA, direct_evolve, walsh
from rcsp_warm.evaluation import labels, metrics, enumerate_routes
from rcsp_warm.protocols import initial, fit, budget

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["tv2_m16_branch_00", "tv2_m16_cycle_00", "tv2_m16_bottleneck_00"]
TOY = dict(id="toy", n=4, s=0, t=3, edges=[[0, 1], [1, 3], [0, 2], [2, 3], [1, 2]], costs=[1, 5, 2, 2, 1], resources=[1, 1, 3, 3, 1], budget=4)

@pytest.mark.parametrize("tid", TASKS)
def test_instance_counts_and_ground_space(tid):
    t = records.task(tid); lab = labels(t); ch = cost_hamiltonian(t)
    assert set(np.flatnonzero(ch.feasible)) == set(lab["feasible_masks"])            # public predicate == independent DFS
    assert set(np.flatnonzero(ch.H == ch.H.min())) == set(lab["optimal_masks"])       # ground space == F*
    assert len(lab["optimal_masks"]) == 1 and ch.H.min() >= -1 - 1e-12 and ch.H.max() <= 1 + 1e-12
    truth = json.loads((ROOT / "data/instances" / f"{tid}.truth.json").read_text())["paths"]
    assert sorted(r["mask"] for r in truth) == sorted(r["mask"] for r in lab["routes"])
    expected = {"tv2_m16_branch_00": (8, 11, 6), "tv2_m16_cycle_00": (8, 18, 11), "tv2_m16_bottleneck_00": (10, 20, 10)}[tid]
    assert (t["n"], lab["route_count"], lab["feasible_count"]) == expected

@pytest.mark.parametrize("tid", TASKS)
@pytest.mark.parametrize("mx", ["X", "MULTI"])
def test_hamiltonian_fingerprints_match_training_arrays(tid, mx):
    import hashlib
    fp = json.loads((ROOT / "data/records/hamiltonian_fingerprints.json").read_text())[f"{tid}__{mx}"]
    t = records.task(tid); ch = cost_hamiltonian(t); eig = spectrum(t, mx)
    assert hashlib.sha256(np.ascontiguousarray(ch.H).tobytes()).hexdigest() == fp["H_sha256"]
    assert hashlib.sha256(np.ascontiguousarray(eig).tobytes()).hexdigest() == fp["eigen_sha256"]

def test_forest_shift_preserves_route_order():
    t = records.task("tv2_m16_cycle_00"); d = forest_transform(t); shifts = set()
    for r in enumerate_routes(t):
        sel = [j for j in range(16) if (r["mask"] >> j) & 1]; shifts.add(sum(d[j] for j in sel) - r["cost"])
    assert len(shifts) == 1

def test_small_dense_reference_and_bit_order():
    ch = cost_hamiltonian(TOY); m = 5; N = 32
    # routes: 0-1-3 (edges 0,1; resource 2) feasible; 0-2-3 (edges 2,3; resource 6) and 0-1-2-3 (edges 0,4,3; resource 5) over budget
    assert ch.feasible[0b00011] and not ch.feasible[0b01100] and not ch.feasible[0b11001]
    assert not ch.feasible[0b00111] and ch.feasible.sum() == 1                          # extra edge / only one feasible route
    X = np.array([[0, 1], [1, 0]]); I = np.eye(2)
    def xs(s):
        out = np.array([[1.0]])
        for q in reversed(range(m)): out = np.kron(out, X if (s >> q) & 1 else I)
        return out
    for mx in ["X", "MULTI"]:
        HM = sum(w * xs(s) for s, w in mixer_terms(TOY, mx)); sim = QAOA(ch.H, spectrum(TOY, mx))
        x = np.random.default_rng(3).uniform(-2, 2, 6); psi = np.ones(N, complex) / np.sqrt(N)
        for k in range(3):
            psi = expm(-1j * x[2 * k + 1] * HM) @ (np.exp(-1j * x[2 * k] * ch.H) * psi)
        assert np.max(abs(sim.evolve(x) - psi)) < 1e-12
        assert np.max(abs(direct_evolve(ch.H, mixer_terms(TOY, mx), x) - psi)) < 1e-12
    assert np.allclose(np.diag(xs(1))[:4], 0) and xs(1)[1, 0] == 1                  # X on qubit 0 flips bit 0

def test_state_norm_zero_layer_and_gradient():
    t, ch, sim, _ = records.build("tv2_m16_bottleneck_00", "MULTI"); x = np.random.default_rng(1).uniform(-1, 1, 6)
    psi = sim.evolve(x); assert abs(np.vdot(psi, psi) - 1) < 1e-12
    assert np.max(abs(sim.evolve(np.r_[x, 0, 0]) - psi)) < 1e-13
    f, g = sim.value_gradient(x); h = 1e-5
    fd = np.array([(sim.value(x + h * e) - sim.value(x - h * e)) / (2 * h) for e in np.eye(6)])
    assert np.max(abs(fd - g)) < 1e-8 and abs(f - sim.value(x)) < 1e-13

def test_metrics_identity():
    t, ch, sim, _ = records.build("tv2_m16_branch_00", "X"); lab = labels(t)
    m = metrics(sim.evolve([0.3, 0.4]), ch.H, lab)
    assert abs(m["p_opt"] - m["p_feas"] * m["p_opt_given_feas"]) < 1e-15
    assert abs(m["mass_optimal"] + m["mass_feasible_nonoptimal"] + m["mass_infeasible"] - 1) < 1e-12

def test_initialisation_reproduces_recorded_vectors():
    A = records.angles(); R = {(r["unit"], int(r["p"]), r["strategy"]): r for r in records.fits()}
    import hashlib
    for u in records.UNITS:
        tid, mx, _ = u.split("__")
        x0, _, _ = initial(tid, mx, 9101, 1, None, "SHARED_P1")
        assert hashlib.sha256(json.dumps(x0.tolist()).encode()).hexdigest() == R[(u, 1, "SHARED_P1")]["initial_parameters_sha256"]
        x0, _, _ = initial(tid, mx, 9101, 120, None, "RANDOM_JOINT")
        assert hashlib.sha256(json.dumps(x0.tolist()).encode()).hexdigest() == R[(u, 120, "RANDOM_JOINT")]["initial_parameters_sha256"]
        for st in ["WARM_JOINT", "FROZEN_LAYERWISE"]:
            for p in [2, 37, 120]:
                par = A[f"{u}/p{p-1:03d}/{'SHARED_P1' if p == 2 else st}"]; x0, anc, pre = initial(tid, mx, 9101, p, par, st)
                full = x0 if pre is None else np.r_[pre, x0]
                assert np.array_equal(full, np.asarray(A[f"{u}/p{p:03d}/{st}#initial"]))
                if st == "WARM_JOINT": assert np.array_equal(anc, np.r_[par, 0, 0])

def test_protocol_rules_small_fit():
    t, ch, sim, _ = records.build("tv2_m16_cycle_00", "X")
    p1 = fit(sim, initial("tv2_m16_cycle_00", "X", 9101, 1, None, "SHARED_P1")[0])
    assert p1["work_credits"] <= budget(1) == 64
    x0, anc, pre = initial("tv2_m16_cycle_00", "X", 9101, 2, p1["angles"], "WARM_JOINT"); w = fit(sim, x0, anchor=anc)
    assert w["active_parameter_count"] == 4 and w["loss"] <= p1["loss"] + 1e-15 and w["work_credits"] <= 64
    x0, anc, pre = initial("tv2_m16_cycle_00", "X", 9101, 2, p1["angles"], "FROZEN_LAYERWISE"); f = fit(sim, x0, anchor=anc, prefix=pre)
    assert np.array_equal(np.asarray(f["angles"])[:2], np.asarray(p1["angles"])) and f["active_parameter_count"] == 2
    assert fit(sim, [0.1, 0.1], credit_budget=2)["work_credits"] == 1                  # <3 credits: one energy-only call

def test_training_modules_do_not_import_evaluation():
    import ast
    for f in ["problem.py", "mixers.py", "simulator.py", "protocols.py"]:
        tree = ast.parse((ROOT / "src/rcsp_warm" / f).read_text())
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not any("evaluation" in m or "records" in m for m in mods), f

def test_verify_records_script():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/verify_records.py")], capture_output=True, text=True)
    assert r.returncode == 0 and '"passed": true' in r.stdout
