"""Generate the README overview figures (docs/assets/*.png and *.svg) from the frozen records.
  python scripts/make_readme_assets.py [--out-dir docs/assets]
Every plotted value is read from data/records/fits.csv; nothing is typed by hand. The script also writes
docs/assets/readme_numbers.json with the headline numbers quoted in README.md (checked by tests)."""
import argparse, json, statistics as st, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rcsp_warm import records

FAM = {"tv2_m16_branch_00": "Multi-branch", "tv2_m16_cycle_00": "Directed-cycle", "tv2_m16_bottleneck_00": "Bottleneck"}
MIX = {"X": "X", "MULTI": "multi-X"}
# Okabe-Ito colours + hatching, so methods are distinguishable in greyscale as well
COL = {"WARM_JOINT": "#0072B2", "RANDOM_JOINT": "#E69F00", "FROZEN_LAYERWISE": "#009E73"}
HAT = {"WARM_JOINT": "", "RANDOM_JOINT": "//", "FROZEN_LAYERWISE": ".."}
NAME = {"WARM_JOINT": "WARM", "RANDOM_JOINT": "RANDOM (single fit)", "FROZEN_LAYERWISE": "FROZEN"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=str(ROOT / "docs/assets")); a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                         "axes.titleweight": "bold", "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10.5, "legend.fontsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none",
                         "hatch.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.12, "savefig.facecolor": "white"})
    R = records.fits(); U = records.UNITS
    g = {(r["unit"], int(r["p"]), r["strategy"]): r for r in R}
    def row(u, p, s): return g[(u, p, "SHARED_P1" if p == 1 else s)]
    def pc(x): return 100 * float(x)
    def lab(u): t, m, _ = u.split("__"); return f"{FAM[t]} / {MIX[m]}"
    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=200); fig.savefig(out / f"{name}.svg"); plt.close(fig)
    N = {}
    # ---------------- A: headline result (linear scale; FROZEN values printed because they are ~0 on this scale)
    order = ["FROZEN_LAYERWISE", "RANDOM_JOINT", "WARM_JOINT"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0)); h = 0.26; ys = np.arange(len(U))[::-1]
    for k, s in enumerate(order):
        vals = [pc(row(u, 120, s)["p_opt"]) for u in U]
        ax.barh(ys + (k - 1) * h, vals, h, color=COL[s], hatch=HAT[s], edgecolor="white", lw=0.6, label=NAME[s])
        for y, v in zip(ys + (k - 1) * h, vals):
            if s == "WARM_JOINT": ax.text(v + 0.8, y, f"{v:.2f}%", va="center", fontsize=10.5, fontweight="bold", color=COL[s])
            elif s == "RANDOM_JOINT": ax.text(v + 0.8, y, f"{v:.2f}%", va="center", fontsize=9, color="#8a5a00")
            else: ax.text(0.8, y, f"{v:.4f}%" if v < 0.01 else f"{v:.3f}%", va="center", fontsize=8.5, color="#006b4f")
    ax.set_yticks(ys); ax.set_yticklabels([lab(u) for u in U]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 62); ax.set_xlabel("optimal-route probability $P_{opt}$ at p = 120 (%)")
    ax.set_title("Optimal-route probability at p = 120", loc="left")
    h_, l_ = ax.get_legend_handles_labels(); ax.legend(h_[::-1], l_[::-1], loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.text(0.0, -0.02, "3 fixed 16-edge RCSP instances × 2 mixers, 1 training seed, exact statevector. Linear scale: FROZEN is < 0.05% everywhere\n"
             "(values printed). RANDOM = one randomly initialised joint fit at p = 120. No error bars: there are no independent repeats.",
             fontsize=8.5, color="0.35", va="top")
    save(fig, "headline_result")
    # ---------------- B: depth progression
    depths = list(range(1, 121)); marks = [16, 32, 64, 100, 120]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for u in U:
        ax.plot(depths, [pc(row(u, p, "WARM_JOINT")["p_opt"]) for p in depths], color=COL["WARM_JOINT"], alpha=0.28, lw=1.2)
    mean = [st.mean(pc(row(u, p, "WARM_JOINT")["p_opt"]) for u in U) for p in depths]
    ax.plot(depths, mean, color="#003f63", lw=3.0)
    for p in marks:
        ax.plot(p, mean[p - 1], "o", color="#003f63", ms=6, zorder=5)
        ax.annotate(f"{mean[p-1]:.2f}%", (p, mean[p - 1]), xytext=(-4, 9), textcoords="offset points", ha="right" if p == 120 else "center", fontsize=10, fontweight="bold", color="#003f63")
        N[f"warm_mean_p{p}"] = round(mean[p - 1], 4)
    ub = "tv2_m16_branch_00__MULTI__s9101"; y112, y120 = pc(row(ub, 112, "WARM_JOINT")["p_opt"]), pc(row(ub, 120, "WARM_JOINT")["p_opt"])
    ax.annotate(f"Multi-branch / multi-X:\n{y112:.2f}% at p=112 → {y120:.2f}% at p=120", xy=(118, y120), xytext=(66, 40),
                fontsize=9, color="0.25", arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.set_xlim(0, 123); ax.set_ylim(0, 60); ax.set_xticks([1, 16, 32, 64, 100, 120])
    ax.set_xlabel("QAOA depth p"); ax.set_ylabel("$P_{opt}$ (%)")
    ax.set_title("WARM success probability grows with continuation depth", loc="left")
    ax.legend(handles=[Line2D([], [], color=COL["WARM_JOINT"], alpha=0.4, lw=1.4, label="individual configurations (6)"),
                       Line2D([], [], color="#003f63", lw=3, label="six-configuration mean")], loc="upper left")
    fig.text(0.0, -0.02, "Depth improves the endpoint trend, but $P_{opt}$ is not monotonic along every trajectory.\n"
             "The six curves are six configurations of one seed, not independent samples.", fontsize=8.5, color="0.35", va="top")
    save(fig, "depth_progression")
    # ---------------- C: probability decomposition (WARM, p = 120)
    fig, ax = plt.subplots(figsize=(7.0, 4.4)); ys = np.arange(len(U))[::-1]
    parts = [("optimal route", "#0072B2", ""), ("feasible, non-optimal", "#9ecae1", "//"), ("infeasible", "#e5e5e5", "..")]
    for u, y in zip(U, ys):
        r = row(u, 120, "WARM_JOINT"); po, pf = pc(r["p_opt"]), pc(r["p_feas"]); left = 0
        for (nm, c, ht), w in zip(parts, [po, pf - po, 100 - pf]):
            ax.barh(y, w, left=left, color=c, hatch=ht, edgecolor="white", lw=0.8, height=0.62); left += w
        ax.text(po / 2, y, f"{po:.2f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(101, y, f"feasible {pf:.2f}%", va="center", fontsize=9.5, color="0.25")
    ax.set_yticks(ys); ax.set_yticklabels([lab(u) for u in U]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("probability mass of the WARM p = 120 state (%)")
    ax.set_title("High feasibility is not the same as high optimal-route probability", loc="left", fontsize=12)
    ax.legend(handles=[Patch(fc=c, hatch=ht, ec="0.6", lw=0.5, label=nm) for nm, c, ht in parts], ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "probability_decomposition")
    # ---------------- D: performance vs cost
    mw = st.mean(pc(row(u, 120, "WARM_JOINT")["p_opt"]) for u in U); mr = st.mean(pc(row(u, 120, "RANDOM_JOINT")["p_opt"]) for u in U)
    lw = st.mean(float(row(u, 120, "WARM_JOINT")["cumulative_layer_work"]) for u in U); lr = st.mean(float(row(u, 120, "RANDOM_JOINT")["cumulative_layer_work"]) for u in U)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for ax, vals, fmt, title, ratio in [(a1, [mr, mw], "{:.2f}%", "Mean $P_{opt}$ at p = 120", mw / mr), (a2, [lr, lw], "{:,.0f}", "Mean cumulative nominal layer-work", lw / lr)]:
        ss = ["RANDOM_JOINT", "WARM_JOINT"]
        ax.bar([0, 1], vals, color=[COL[s] for s in ss], hatch=[HAT[s] for s in ss], edgecolor="white", width=0.6)
        for x, v in zip([0, 1], vals): ax.text(x, v * 1.02, fmt.format(v), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["RANDOM\n(single fit)", "WARM"]); ax.set_title(title, fontsize=11.5)
        ax.set_ylim(0, max(vals) * 1.25); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.text(0.5, 0.93, f"≈ {ratio:.2f}×" if ax is a1 else f"≈ {ratio:.1f}×", transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold", color="0.25")
    fig.suptitle("Higher endpoint success comes with much more training work", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, -0.04, "Means over the six configurations at p = 120. WARM work includes every fit of its p = 1…120 chain; RANDOM is one fit.\n"
             "Not budget-matched: this is an endpoint comparison, not a training-efficiency or speed-up result.", fontsize=8.5, color="0.35", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "performance_cost")
    w120 = [pc(row(u, 120, "WARM_JOINT")["p_opt"]) for u in U]
    N.update(warm_p120_mean=round(mw, 4), warm_p120_min=round(min(w120), 2), warm_p120_max=round(max(w120), 2), warm_p120_median=round(st.median(w120), 2),
             random_p120_mean=round(mr, 4), warm_layer_work_mean=lw, random_layer_work_mean=lr, work_ratio=round(lw / lr, 2), popt_ratio=round(mw / mr, 2),
             warm_beats_frozen=sum(pc(row(u, 120, "WARM_JOINT")["p_opt"]) > pc(row(u, 120, "FROZEN_LAYERWISE")["p_opt"]) for u in U),
             warm_beats_random=sum(pc(row(u, 120, "WARM_JOINT")["p_opt"]) > pc(row(u, 120, "RANDOM_JOINT")["p_opt"]) for u in U),
             p120={lab(u): {s: round(pc(row(u, 120, s)["p_opt"]), 6) for s in order} | {"WARM_p_feas": round(pc(row(u, 120, "WARM_JOINT")["p_feas"]), 2)} for u in U})
    (out / "readme_numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False) + "\n"); print(json.dumps(N, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
