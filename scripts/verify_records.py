"""Level-1 check: internal consistency of the frozen records (no simulation).
  python scripts/verify_records.py"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcsp_warm import records
from rcsp_warm.protocols import budget

def main():
    R = records.fits(); A = records.angles(); problems = []; n = {}
    def ok(name, cond, detail=""):
        n[name] = n.get(name, 0) + 1
        if not cond: problems.append(f"{name}: {detail}")
    idx = {(r["unit"], int(r["p"]), r["strategy"]): r for r in R}
    ok("row_count_1470", len(R) == 1470, len(R))
    for u in records.UNITS:
        ok("shared_p1_once", sum(1 for r in R if r["unit"] == u and r["p"] == "1") == 1, u)
        for st in ["FROZEN_LAYERWISE", "WARM_JOINT"]:
            ok("complete_chain", all((u, p, st) in idx for p in range(2, 121)), (u, st))
        ok("random_checkpoints", sorted(int(r["p"]) for r in R if r["unit"] == u and r["strategy"] == "RANDOM_JOINT") == [4, 8, 16, 32, 64, 120], u)
    for (u, p, st), r in idx.items():
        key = f"{u}/p{p:03d}/{st}"; x = np.asarray(A[key])
        ok("angle_length_2p", len(x) == 2 * p, key)
        ok("angles_in_box", np.all(np.abs(x) <= 4), key)
        ok("budget_respected", int(r["work_credits"]) <= int(r["work_budget"]) == budget(p), key)
        ok("credit_rule", int(r["work_credits"]) == (int(r["objective_evaluations"]) - int(r["gradient_evaluations"])) + 3 * int(r["gradient_evaluations"]), key)
        ok("layer_work_rule", int(r["nominal_layer_work"]) == p * int(r["work_credits"]), key)
        ok("cumulative_rule", int(r["cumulative_layer_work"]) == int(r["ancestor_layer_work"]) + int(r["nominal_layer_work"]), key)
        ok("per_state_identity_popt_le_pfeas", float(r["p_opt"]) <= float(r["p_feas"]) + 1e-15, key)
        if st == "RANDOM_JOINT" or p == 1:
            ok("no_ancestor_cost", int(r["ancestor_layer_work"]) == 0, key)
        else:
            par = idx[(u, p - 1, "SHARED_P1" if p == 2 else st)]
            ok("parent_hash_chain", r["parent_sha256"] == par["winner_sha256"], key)
            ok("ancestor_equals_parent_cumulative", int(r["ancestor_layer_work"]) == int(par["cumulative_layer_work"])
               and int(r["ancestor_credits"]) == int(par["cumulative_credits"]), key)
            pa = np.asarray(A[f"{u}/p{p-1:03d}/{'SHARED_P1' if p == 2 else st}"]); init = np.asarray(A[key + "#initial"])
            ok("inherits_parent_prefix", np.array_equal(init[:-2], pa), key)
            if st == "FROZEN_LAYERWISE":
                ok("frozen_prefix_unchanged", np.array_equal(x[:-2], pa) and int(r["active_parameter_count"]) == 2, key)
            else:
                ok("warm_all_parameters_active", int(r["active_parameter_count"]) == 2 * p, key)
    rep = dict(passed=not problems, checks=n, problems=problems[:20])
    print(json.dumps(rep, indent=1)); return 0 if not problems else 1

if __name__ == "__main__":
    sys.exit(main())
