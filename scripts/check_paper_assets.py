"""Compare regenerated assets with the frozen reference outputs shipped in reference_outputs/.
  python scripts/check_paper_assets.py --shipped reference_outputs --regenerated reproduced_assets

Strict (byte-identical): every file in tables/, and every entry of generated/numbers.tex and numbers.json
except the finite-difference gradient diagnostic `GradErr`.

`GradErr` is max|adjoint - central finite difference| (h = 1e-5) at p = 3 on the six configurations. It is a
floating-point round-off quantity that differs between CPUs/BLAS builds (9.6e-11 on the study machine,
1.1e-10 on a GitHub runner). Its scale is ~eps*|E|/h ~ 1e-11 plus O(h^2) truncation; a gradient bug would give
errors of order the gradient itself (1e-2 to 1).
The finite-difference diagnostic is platform-sensitive; the threshold 5e-10 is several times above the observed
~1e-10 numerical error (3.1e-11 to 1.1e-10 across machines and runs) while remaining far below any scientifically
relevant scale. Only this one diagnostic is exempt from byte-identical comparison.
"""
import argparse, filecmp, json, re, sys
from pathlib import Path

PLATFORM_SENSITIVE = {"GradErr"}
GRAD_ERR_BOUND = 5e-10

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shipped", required=True); ap.add_argument("--regenerated", required=True); a = ap.parse_args()
    S, G = Path(a.shipped), Path(a.regenerated); problems = []
    # tables: strict
    st, gt = sorted(p.name for p in (S / "tables").iterdir()), sorted(p.name for p in (G / "tables").iterdir())
    if st != gt: problems.append(f"table file sets differ: {st} vs {gt}")
    for n in set(st) & set(gt):
        if not filecmp.cmp(S / "tables" / n, G / "tables" / n, shallow=False): problems.append(f"table differs: {n}")
    # numbers.json: strict except platform-sensitive keys
    sj, gj = json.loads((S / "generated/numbers.json").read_text()), json.loads((G / "generated/numbers.json").read_text())
    if set(sj) != set(gj): problems.append(f"numbers.json keys differ: {set(sj) ^ set(gj)}")
    for k in set(sj) & set(gj) - PLATFORM_SENSITIVE:
        if sj[k] != gj[k]: problems.append(f"number differs: {k}: {sj[k]} -> {gj[k]}")
    # numbers.tex: strict line-by-line except the platform-sensitive macros
    def lines(p): return [l for l in p.read_text().splitlines() if not any(f"\\num{k}}}" in l for k in PLATFORM_SENSITIVE)]
    if lines(S / "generated/numbers.tex") != lines(G / "generated/numbers.tex"): problems.append("numbers.tex differs outside GradErr")
    # GradErr: absolute bound
    m = re.fullmatch(r"([0-9.]+)\\times10\^\{(-?\d+)\}", gj.get("GradErr", ""))
    grad = float(m.group(1)) * 10 ** int(m.group(2)) if m else float("nan")
    if not grad < GRAD_ERR_BOUND: problems.append(f"GradErr {gj.get('GradErr')} not below {GRAD_ERR_BOUND}")
    print(json.dumps(dict(passed=not problems, tables_compared=len(set(st) & set(gt)), numbers_compared_strict=len(set(sj) & set(gj) - PLATFORM_SENSITIVE),
                          grad_err_shipped=sj.get("GradErr"), grad_err_regenerated=gj.get("GradErr"), grad_err_value=grad, grad_err_bound=GRAD_ERR_BOUND,
                          problems=problems), indent=1))
    return 0 if not problems else 1

if __name__ == "__main__":
    sys.exit(main())
