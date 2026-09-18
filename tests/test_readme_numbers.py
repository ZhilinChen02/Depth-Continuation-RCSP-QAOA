"""The headline numbers quoted in README.md must be the values computed from data/records."""
import csv, statistics as st
from pathlib import Path
from rcsp_warm import records

ROOT = Path(__file__).resolve().parents[1]

def test_readme_headline_numbers_match_records():
    readme = (ROOT / "README.md").read_text()
    R = {(r["unit"], int(r["p"]), r["strategy"]): r for r in records.fits()}; U = records.UNITS
    pw = [100 * float(R[(u, 120, "WARM_JOINT")]["p_opt"]) for u in U]
    pr = [100 * float(R[(u, 120, "RANDOM_JOINT")]["p_opt"]) for u in U]
    pf = [100 * float(R[(u, 120, "FROZEN_LAYERWISE")]["p_opt"]) for u in U]
    lw = st.mean(float(R[(u, 120, "WARM_JOINT")]["cumulative_layer_work"]) for u in U)
    lr = st.mean(float(R[(u, 120, "RANDOM_JOINT")]["cumulative_layer_work"]) for u in U)
    expected = [f"**{st.mean(pw):.2f}%**", f"**{min(pw):.2f}–{max(pw):.2f}%**", f"{st.median(pw):.2f}%",
                f"**{lw / lr:.1f}×**", f"**{st.mean(pw) / st.mean(pr):.2f}×**", f"{lw:,.0f}", f"{lr:,.0f}"]
    for p in [16, 32, 64, 100, 120]:
        expected.append(f"{st.mean(100 * float(R[(u, p, 'WARM_JOINT')]['p_opt']) for u in U):.2f}%")
    for s in expected:
        assert s in readme, s
    assert all(w > f and w > r for w, f, r in zip(pw, pf, pr)) and "**6 / 6 higher**" in readme
    assert max(pf) < 0.05 and "below 0.05%" in readme
