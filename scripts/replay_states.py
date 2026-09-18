"""Level-2 check: rebuild each Hamiltonian from the instance definition, evolve the frozen winner angles
with the numpy simulator, and compare P_opt, P_feas and energy with the recorded values.
  python scripts/replay_states.py --scope endpoints      # the 18 p=120 endpoints (~1 min)
  python scripts/replay_states.py --scope checkpoints    # p in {1,4,8,16,32,64,120}, all strategies
  python scripts/replay_states.py --scope all            # all 1470 fits (~30-60 CPU-min)
Also writes the p=120 route distribution reported in the paper (results/p120_route_distribution.csv)."""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rcsp_warm import records
from rcsp_warm.evaluation import labels, metrics
from rcsp_warm.simulator import direct_evolve

TOL = 1e-9
CHK = {1, 4, 8, 16, 32, 64, 120}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scope", choices=["endpoints", "checkpoints", "all"], default="endpoints")
    ap.add_argument("--out", default=str(ROOT / "results")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = records.fits(); A = records.angles(); worst = dict(p_opt=0.0, p_feas=0.0, energy=0.0); n = 0; fails = []; dist = []; indep = []
    t0 = time.time()
    for u in records.UNITS:
        tid, mx, _ = u.split("__"); task, ch, sim, terms = records.build(tid, mx); lab = labels(task)
        for r in R:
            p = int(r["p"])
            if r["unit"] != u: continue
            if a.scope == "endpoints" and p != 120: continue
            if a.scope == "checkpoints" and p not in CHK: continue
            x = A[f"{u}/p{p:03d}/{r['strategy']}"]; psi = sim.evolve(x); m = metrics(psi, ch.H, lab); n += 1
            for k, rec in [("p_opt", "p_opt"), ("p_feas", "p_feas"), ("energy", "loss")]:
                d = abs(m[k] - float(r[rec])); worst[k] = max(worst[k], d)
                if d > TOL: fails.append((u, p, r["strategy"], k, d))
            if p == 120:
                pr = np.abs(psi) ** 2; feas = sorted(lab["feasible_masks"], key=lambda s: -pr[s]); opt = set(lab["optimal_masks"])
                cost = {rr["mask"]: rr["cost"] for rr in lab["routes"]}
                for rank, s in enumerate(feas):
                    dist.append(dict(unit=u, strategy=r["strategy"], rank=rank + 1, mask=s, cost=cost[s], optimal=s in opt, probability=float(pr[s])))
                if r["strategy"] == "WARM_JOINT":
                    indep.append(float(np.max(np.abs(direct_evolve(ch.H, terms, x) - psi))))
    with open(out / "p120_route_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist[0])); w.writeheader(); w.writerows(dist)
    rep = dict(scope=a.scope, fits_replayed=n, tolerance=TOL, max_abs_diff=worst, failures=fails[:20], passed=not fails,
               independent_direct_rotation_max_diff_p120_warm=max(indep) if indep else None, seconds=time.time() - t0)
    (out / f"REPLAY_{a.scope}.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
