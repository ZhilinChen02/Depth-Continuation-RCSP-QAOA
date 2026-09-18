"""Re-execute the training protocols (explicit opt-in; never started automatically).
  python scripts/train.py --task tv2_m16_bottleneck_00 --mixer MULTI --max-depth 3 --out results/smoke
Full retraining to p=120 requires --confirm-full-retraining and takes several CPU-hours per configuration.
Reproduces the protocol, not bit-identical trajectories: the original run used Numba kernels, and floating-point
differences can change L-BFGS-B paths. Compare against data/records, do not overwrite it."""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rcsp_warm import records
from rcsp_warm.protocols import initial, fit

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--mixer", choices=["X", "MULTI"], required=True)
    ap.add_argument("--seed", type=int, default=9101); ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--strategies", default="WARM_JOINT,FROZEN_LAYERWISE"); ap.add_argument("--out", default=str(ROOT / "results/train"))
    ap.add_argument("--confirm-full-retraining", action="store_true"); a = ap.parse_args()
    if a.max_depth > 8 and not a.confirm_full_retraining:
        sys.exit("max-depth > 8 is a long run; pass --confirm-full-retraining explicitly.")
    task, ch, sim, _ = records.build(a.task, a.mixer); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    x0, _, _ = initial(a.task, a.mixer, a.seed, 1, None, "SHARED_P1"); t = time.time(); p1 = fit(sim, x0); p1["seconds"] = time.time() - t
    rows.append(dict(strategy="SHARED_P1", **{k: p1[k] for k in ["p", "loss", "status", "work_credits", "nominal_layer_work", "seconds"]}))
    for st in a.strategies.split(","):
        parent = p1
        for p in range(2, a.max_depth + 1):
            x0, anc, pre = initial(a.task, a.mixer, a.seed, p, parent["angles"], st); t = time.time()
            rec = fit(sim, x0, anchor=anc, prefix=pre); rec["seconds"] = time.time() - t
            rows.append(dict(strategy=st, **{k: rec[k] for k in ["p", "loss", "status", "work_credits", "nominal_layer_work", "seconds"]}))
            (out / f"{a.task}__{a.mixer}__{st}__p{p:03d}.json").write_text(json.dumps(rec)); parent = rec
    with open(out / f"{a.task}__{a.mixer}__summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows, indent=1))

if __name__ == "__main__":
    main()
