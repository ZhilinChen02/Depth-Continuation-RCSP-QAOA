"""Generate every figure (vector PDF), table and number macro reported in the accompanying paper from the frozen records.
  python scripts/make_paper_assets.py --out-dir reproduced_assets
Writes figures/, tables/ and generated/ into any (new or existing) output directory; no manuscript source is needed.
No value is typed by hand; the manuscript reads generated/numbers.tex and tables/*.tex."""
import argparse, csv, json, statistics as st, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rcsp_warm import records
from rcsp_warm.evaluation import labels
from rcsp_warm.mixers import graph_masks

FAM = {"tv2_m16_branch_00": "Multi-branch", "tv2_m16_cycle_00": "Directed-cycle", "tv2_m16_bottleneck_00": "Bottleneck"}
MIX = {"X": "X", "MULTI": "multi-X"}
STRAT = [("WARM_JOINT", "WARM"), ("RANDOM_JOINT", "RANDOM"), ("FROZEN_LAYERWISE", "FROZEN")]
CHK = [1, 4, 8, 16, 32, 64, 120]

def label(u):
    t, m, _ = u.split("__"); return f"{FAM[t]} / {MIX[m]}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", "--paper-dir", dest="out_dir", default=str(ROOT / "reproduced_assets")); a = ap.parse_args()
    P = Path(a.out_dir); [ (P / d).mkdir(parents=True, exist_ok=True) for d in ["figures", "tables", "generated"] ]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    R = records.fits(); U = records.UNITS
    g = {(r["unit"], int(r["p"]), r["strategy"]): r for r in R}
    def row(u, p, s): return g[(u, p, "SHARED_P1" if p == 1 else s)]
    def pc(x): return 100 * float(x)
    # Okabe-Ito colour-blind-safe palette; shapes also differ, so no information is carried by colour alone
    C = {"WARM_JOINT": "#0072B2", "RANDOM_JOINT": "#D55E00", "FROZEN_LAYERWISE": "#009E73"}
    MK = {"WARM_JOINT": "o", "RANDOM_JOINT": "s", "FROZEN_LAYERWISE": "^"}
    NAME = dict(STRAT)
    W_IN = 6.3                                   # = LaTeX \textwidth (16 cm), figures are included at 100 %
    def logfmt(ax, axis="y"):
        from matplotlib.ticker import LogLocator, FuncFormatter
        f = FuncFormatter(lambda v, _: f"$10^{{{int(np.round(np.log10(v)))}}}$")
        getattr(ax, f"{axis}axis").set_major_locator(LogLocator(base=10, numticks=12)); getattr(ax, f"{axis}axis").set_major_formatter(f)
    def grid(ax, which="y"): ax.grid(axis=which, color="0.88", lw=0.5); ax.set_axisbelow(True)
    # ---- Figure 1: p=120 P_opt, dot plot on a log axis (no bars: a bar length on a log axis has no meaning)
    fig, ax = plt.subplots(figsize=(W_IN, 2.75))
    ys = np.arange(len(U))[::-1]
    for u, y in zip(U, ys):
        vals = [pc(row(u, 120, s)["p_opt"]) for s, _ in STRAT]
        ax.plot([min(vals), max(vals)], [y, y], color="0.8", lw=0.8, zorder=1)
        for (s, _), v in zip(STRAT, vals):
            ax.plot(v, y, MK[s], color=C[s], ms=5.5, mec="white", mew=0.5, zorder=3)
            txt = f"{v:.3g}" if v < 1 else f"{v:.1f}"
            below = s == "RANDOM_JOINT"
            ax.annotate(txt, (v, y), xytext=(0, -6 if below else 5.5), textcoords="offset points", ha="center",
                        va="top" if below else "bottom", fontsize=6.3, color=C[s])
    ax.set_xscale("log"); ax.set_xlim(1.5e-4, 250); logfmt(ax, "x"); grid(ax, "x")
    ax.axvline(100 / 2 ** 16, color="0.45", lw=0.7, ls=":", zorder=0)
    ax.text(100 / 2 ** 16 * 1.12, len(U) - 0.45, "uniform state\n($2^{-16}$)", fontsize=6.3, color="0.35", va="top", ha="left")
    ax.set_yticks(ys); ax.set_yticklabels([label(u) for u in U]); ax.set_ylim(-0.6, len(U) - 0.25)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"$P_{\mathrm{opt}}$ at $p=120$ (%, logarithmic)")
    ax.legend(handles=[Line2D([], [], ls="none", marker=MK[s], color=C[s], mec="white", ms=6, label=n) for s, n in STRAT],
              ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.5)
    fig.savefig(P / "figures/fig1_p120_popt.pdf"); plt.close(fig)
    # ---- Figures 2-4: 2 x 3 line-chart small multiples (columns = instance, rows = mixer); every protocol is a line.
    # RANDOM exists only at p in {4,8,16,32,64,120}: its dash-dot line joins those six independent fits (segments are not data).
    LS = {"WARM_JOINT": dict(color=C["WARM_JOINT"], lw=1.4, ls="-"), "FROZEN_LAYERWISE": dict(color=C["FROZEN_LAYERWISE"], lw=1.1, ls=(0, (3, 1.5))),
          "RANDOM_JOINT": dict(color=C["RANDOM_JOINT"], lw=1.1, ls=(0, (5, 1.5, 1, 1.5)))}
    RLABEL = {"WARM_JOINT": "WARM", "FROZEN_LAYERWISE": "FROZEN", "RANDOM_JOINT": "RANDOM (joins single fits at $p$ = 4, 8, 16, 32, 64, 120)"}
    def depths(s): return list(range(1, 121)) if s != "RANDOM_JOINT" else CHK[1:]
    def small_multiples(fname, draw, ylabel, ylog, xlabel, xlog, legend):
        fig, axs = plt.subplots(2, 3, figsize=(W_IN, 3.95), sharex=True, sharey=True)
        for j, t in enumerate(FAM):
            for i, mx in enumerate(["X", "MULTI"]):
                ax = axs[i, j]; u = f"{t}__{mx}__s9101"; draw(ax, u)
                if ylog: ax.set_yscale("log"); ax.set_ylim(5e-5, 150); logfmt(ax, "y")
                else: ax.set_ylim(0, 1.02)
                if xlog: ax.set_xscale("log"); logfmt(ax, "x")
                else: ax.set_xlim(0, 122); ax.set_xticks([0, 32, 64, 96, 120])
                grid(ax, "both")
                if i == 0: ax.set_title(FAM[t], pad=4)
                if i == 1: ax.set_xlabel(xlabel)
                if j == 0: ax.set_ylabel(("X mixer" if mx == "X" else "multi-X mixer") + "\n" + ylabel)
                ax.text(0.03, 0.95, "abcdef"[3 * i + j], transform=ax.transAxes, va="top", fontsize=8)
        fig.legend(handles=legend, ncol=len(legend), loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.2 if len(legend) > 3 else 1.6, handlelength=2.6 if len(legend) > 3 else 3.2)
        fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8, w_pad=0.6); fig.savefig(P / f"figures/{fname}"); plt.close(fig)
    proto_legend = [Line2D([], [], label=RLABEL[s], **LS[s]) for s in ["WARM_JOINT", "FROZEN_LAYERWISE", "RANDOM_JOINT"]]
    def draw_popt(xkey):
        def f(ax, u):
            for s in ["FROZEN_LAYERWISE", "RANDOM_JOINT", "WARM_JOINT"]:
                xs = [p if xkey == "p" else float(row(u, p, s)["cumulative_layer_work"]) for p in depths(s)]
                ax.plot(xs, [pc(row(u, p, s)["p_opt"]) for p in depths(s)], **LS[s])
        return f
    small_multiples("fig2_trajectories.pdf", draw_popt("p"), r"$P_{\mathrm{opt}}$ (%)", True, "depth $p$", False, proto_legend)
    small_multiples("fig4_popt_vs_work.pdf", draw_popt("lw"), r"$P_{\mathrm{opt}}$ (%)", True, "cumulative layer-work", True, proto_legend)
    def draw_mass(ax, u):
        for s in ["FROZEN_LAYERWISE", "RANDOM_JOINT", "WARM_JOINT"]:
            ps = depths(s); pf = [float(row(u, p, s)["p_feas"]) for p in ps]; po = [float(row(u, p, s)["p_opt"]) for p in ps]
            ax.plot(ps, pf, color=C[s], lw=1.4, ls="-")
            ax.plot(ps, [o / f if f > 0 else np.nan for o, f in zip(po, pf)], color=C[s], lw=1.2, ls=(0, (1, 1.3)))
    mass_legend = [Line2D([], [], color=C[s], lw=1.4, label=n) for s, n in [("WARM_JOINT", "WARM"), ("FROZEN_LAYERWISE", "FROZEN"), ("RANDOM_JOINT", "RANDOM (6 fits)")]] + \
                  [Line2D([], [], color="0.3", lw=1.4, ls="-", label=r"solid: $P_{\mathrm{feas}}$"), Line2D([], [], color="0.3", lw=1.2, ls=(0, (1, 1.3)), label=r"dotted: $P(\mathrm{opt}\mid\mathrm{feas})$")]
    small_multiples("fig3_mass_decomposition.pdf", draw_mass, "probability", False, "depth $p$", False, mass_legend)
    # ---- numbers
    N = {}
    def put(k, v): N[k] = v
    W = lambda s, p: [float(row(u, p, s)["p_opt"]) for u in U]
    for p in [16, 32, 64, 100, 120]: put(f"WarmMeanP{p}", f"{100*st.mean(W('WARM_JOINT', p)):.2f}")
    put("WarmMinPonetwenty", f"{100*min(W('WARM_JOINT',120)):.2f}"); put("WarmMaxPonetwenty", f"{100*max(W('WARM_JOINT',120)):.2f}")
    put("RandomMeanPonetwenty", f"{100*st.mean(W('RANDOM_JOINT',120)):.2f}"); put("FrozenMeanPonetwenty", f"{100*st.mean(W('FROZEN_LAYERWISE',120)):.4f}")
    put("RandomMinPonetwenty", f"{100*min(W('RANDOM_JOINT',120)):.2f}"); put("RandomMaxPonetwenty", f"{100*max(W('RANDOM_JOINT',120)):.2f}")
    put("FrozenMaxPonetwenty", f"{100*max(W('FROZEN_LAYERWISE',120)):.3f}")
    put("RatioWarmRandom", f"{st.mean(W('WARM_JOINT',120))/st.mean(W('RANDOM_JOINT',120)):.2f}")
    lw = lambda s: st.mean(float(row(u, 120, s)["cumulative_layer_work"]) for u in U); sec = lambda s: st.mean(float(row(u, 120, s)["cumulative_fit_seconds"]) for u in U) / 60
    put("LWWarm", f"{lw('WARM_JOINT'):,.0f}".replace(",", "{,}")); put("LWRandom", f"{lw('RANDOM_JOINT'):,.0f}".replace(",", "{,}")); put("LWFrozen", f"{lw('FROZEN_LAYERWISE'):,.1f}".replace(",", "{,}"))
    put("RatioLW", f"{lw('WARM_JOINT')/lw('RANDOM_JOINT'):.2f}")
    put("MinWarm", f"{sec('WARM_JOINT'):.1f}"); put("MinRandom", f"{sec('RANDOM_JOINT'):.2f}"); put("MinFrozen", f"{sec('FROZEN_LAYERWISE'):.2f}")
    tr = ch = both = edown = 0; drops = []
    for u in U:
        for p in range(2, 121):
            a_, b_ = row(u, p, "WARM_JOINT"), row(u, p - 1, "WARM_JOINT"); tr += 1
            ch += float(a_["earlier_parameter_change"]) > 0
            ed = float(a_["loss"]) < float(b_["loss"]); edown += ed
            if ed and float(a_["p_opt"]) < float(b_["p_opt"]): both += 1
    put("Transitions", tr); put("TransitionsChanged", ch); put("TransitionsEdown", edown); put("TransitionsEdownPdown", both)
    ub = "tv2_m16_branch_00__MULTI__s9101"
    put("BranchMultiPeak", f"{pc(row(ub,112,'WARM_JOINT')['p_opt']):.2f}"); put("BranchMultiEnd", f"{pc(row(ub,120,'WARM_JOINT')['p_opt']):.2f}")
    put("BranchMultiEnergyPeak", f"{float(row(ub,112,'WARM_JOINT')['loss']):.4f}"); put("BranchMultiEnergyEnd", f"{float(row(ub,120,'WARM_JOINT')['loss']):.4f}")
    D = list(csv.DictReader(open(ROOT / "data/records/p120_route_distribution.csv")))
    bm = [d for d in D if d["unit"] == "tv2_m16_bottleneck_00__MULTI__s9101" and d["strategy"] == "WARM_JOINT"]
    put("BottleMultiFirst", f"{100*float(bm[0]['probability']):.2f}"); put("BottleMultiSecond", f"{100*float(bm[1]['probability']):.2f}")
    put("BottleMultiTopTwo", f"{100*(float(bm[0]['probability'])+float(bm[1]['probability'])):.2f}")
    fr = [r for r in R if r["strategy"] == "FROZEN_LAYERWISE" and int(r["p"]) == 120]
    put("FrozenConvergedEnd", sum(r["status"] == "CONVERGED" for r in fr))
    put("WarmBudgetFits", sum(r["status"] == "BUDGET_EXHAUSTED" for r in R if r["strategy"] == "WARM_JOINT"))
    put("FrozenConvergedFits", sum(r["status"] == "CONVERGED" for r in R if r["strategy"] == "FROZEN_LAYERWISE"))
    put("CnotMin", f"{min(int(r['cnot']) for r in R if int(r['p'])==120)/1e6:.3f}"); put("CnotMax", f"{max(int(r['cnot']) for r in R if int(r['p'])==120)/1e6:.3f}")
    put("AncMin", min(int(r['ancillas']) for r in R if int(r['p'])==120)); put("AncMax", max(int(r['ancillas']) for r in R if int(r['p'])==120))
    put("PhysicalFits", len(R))
    wins = sum(float(row(u,120,'WARM_JOINT')['p_opt']) > max(float(row(u,120,'RANDOM_JOINT')['p_opt']), float(row(u,120,'FROZEN_LAYERWISE')['p_opt'])) for u in U)
    put("WarmWinsEnd", wins)
    pf = lambda s: [float(row(u,120,s)['p_feas']) for u in U]; cf = lambda s: [float(row(u,120,s)['p_opt'])/float(row(u,120,s)['p_feas']) for u in U]
    put("WarmPfeasMin", f"{100*min(pf('WARM_JOINT')):.1f}"); put("WarmPfeasMax", f"{100*max(pf('WARM_JOINT')):.1f}")
    put("RandomPfeasMin", f"{100*min(pf('RANDOM_JOINT')):.1f}"); put("RandomPfeasMax", f"{100*max(pf('RANDOM_JOINT')):.1f}")
    put("FrozenPfeasMax", f"{100*max(pf('FROZEN_LAYERWISE')):.2f}")
    put("WarmCondMin", f"{100*min(cf('WARM_JOINT')):.1f}"); put("WarmCondMax", f"{100*max(cf('WARM_JOINT')):.1f}")
    put("RandomCondMin", f"{100*min(cf('RANDOM_JOINT')):.1f}"); put("RandomCondMax", f"{100*max(cf('RANDOM_JOINT')):.1f}")
    # ---- Table 1: instances
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule", r"Instance & $n$ & $m$ & budget & simple $s$--$t$ routes & feasible routes & $|F_\ast|$ & $K$ (multi-X masks) \\", r"\midrule"]
    for t in FAM:
        task = records.task(t); lab = labels(task)
        lines.append(f"{FAM[t]} & {task['n']} & {len(task['edges'])} & {task['budget']} & {lab['route_count']} & {lab['feasible_count']} & {len(lab['optimal_masks'])} & {len(graph_masks(task))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]; (P / "tables/table1_instances.tex").write_text("\n".join(lines) + "\n")
    # ---- Table 2: endpoint results and cost
    lines = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
             r"Configuration & Protocol & $P_{\mathrm{opt}}$ & $P_{\mathrm{feas}}$ & $P(\mathrm{opt}\mid\mathrm{feas})$ & layer-work & fit min & status \\", r"\midrule"]
    for u in U:
        for n_, (s, name) in enumerate(STRAT):
            r = row(u, 120, s); po, pf_ = float(r["p_opt"]), float(r["p_feas"])
            lines.append(f"{label(u) if n_ == 0 else ''} & {name} & {100*po:.4g} & {100*pf_:.4g} & {100*po/pf_:.3g} & {int(r['cumulative_layer_work']):,} & {float(r['cumulative_fit_seconds'])/60:.2f} & {'budget' if r['status']=='BUDGET_EXHAUSTED' else 'conv.'} \\\\".replace(",", "{,}"))
        lines.append(r"\addlinespace" if u != U[-1] else "")
    lines += [r"\bottomrule", r"\end{tabular}"]; (P / "tables/table2_endpoints.tex").write_text("\n".join(lines) + "\n")
    # ---- Supplement tables: checkpoints; transitions per unit; resources
    lines = [r"\begin{longtable}{llrrrr}", r"\toprule", r"Configuration & $p$ & WARM & FROZEN & RANDOM & WARM energy \\", r"\midrule\endhead"]
    for u in U:
        for p in CHK:
            lines.append(f"{label(u) if p == 1 else ''} & {p} & {pc(row(u,p,'WARM_JOINT')['p_opt']):.4g} & {pc(row(u,p,'FROZEN_LAYERWISE')['p_opt']):.4g} & {pc(row(u,p,'RANDOM_JOINT')['p_opt']):.4g} & {float(row(u,p,'WARM_JOINT')['loss']):.5f} \\\\")
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{longtable}"]; (P / "tables/tableS1_checkpoints.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Configuration & transitions & $E\downarrow$ & $E\downarrow$ and $P_{\mathrm{opt}}\downarrow$ & largest drop (pp) & at $p$ \\", r"\midrule"]
    for u in U:
        e = both_u = 0; big = (0.0, None)
        for p in range(2, 121):
            a_, b_ = row(u, p, "WARM_JOINT"), row(u, p - 1, "WARM_JOINT"); ed = float(a_["loss"]) < float(b_["loss"]); e += ed
            dp = pc(a_["p_opt"]) - pc(b_["p_opt"])
            if ed and dp < 0: both_u += 1
            if dp < big[0]: big = (dp, p)
        lines.append(f"{label(u)} & 119 & {e} & {both_u} & {big[0]:.2f} & {big[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]; (P / "tables/tableS2_transitions.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Configuration & CNOT & Toffoli & one-qubit & work ancillas \\", r"\midrule"]
    for u in U:
        r = row(u, 120, "WARM_JOINT"); lines.append(f"{label(u)} & {int(r['cnot']):,} & {int(r['toffoli']):,} & {int(r['one_qubit']):,} & {r['ancillas']} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]; (P / "tables/tableS3_resources.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{llrrrl}", r"\toprule", r"Configuration & Protocol & rank & cost & optimal & probability (\%) \\", r"\midrule"]
    for u in U:
        for s, name in STRAT:
            dd = [d for d in D if d["unit"] == u and d["strategy"] == s][:2]
            for d in dd:
                lines.append(f"{label(u) if (s=='WARM_JOINT' and d['rank']=='1') else ''} & {name if d['rank']=='1' else ''} & {d['rank']} & {d['cost']} & {'yes' if d['optimal']=='True' else 'no'} & {100*float(d['probability']):.4g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]; (P / "tables/tableS4_top_routes.tex").write_text("\n".join(lines) + "\n")
    # ---- gradient check (adjoint vs central finite differences, h = 1e-5) on each instance/mixer, p = 3
    errs = []
    for u in U:
        tid, mx, _ = u.split("__"); _, ch_, sim, _ = records.build(tid, mx)
        x = np.random.default_rng(7).uniform(-1, 1, 6); f0, gr = sim.value_gradient(x); fd = np.zeros(6)
        for k in range(6):
            e = np.zeros(6); e[k] = 1e-5; fd[k] = (sim.value(x + e) - sim.value(x - e)) / 2e-5
        errs.append(float(np.max(np.abs(fd - gr))))
    mant, ex = f"{max(errs):.1e}".split("e"); put("GradErr", mant + r"\times10^{" + str(int(ex)) + "}")
    # ---- Table S5: multi-X masks
    lines = [r"\begin{longtable}{p{2.3cm}p{12.6cm}}", r"\toprule", r"Instance & masks $S$ (edge indices $j$, in mixer order) \\", r"\midrule\endhead"]
    for t in FAM:
        ms = graph_masks(records.task(t))
        txt = "; ".join("\\{" + ",".join(str(j) for j in range(16) if (s_ >> j) & 1) + "\\}" for s_ in ms)
        lines.append(f"{FAM[t]} & {txt} \\\\ \\addlinespace")
    lines += [r"\bottomrule", r"\end{longtable}"]; (P / "tables/tableS5_masks.tex").write_text("\n".join(lines) + "\n")
    # ---- Table S6: instance data
    lines = [r"\begin{longtable}{lrrrr}", r"\toprule", r"Instance & edge $j$ & $(u,v)$ & cost $c_j$ & resource $r_j$ \\", r"\midrule\endhead"]
    for t in FAM:
        task = records.task(t)
        for j, (e, c_, r_) in enumerate(zip(task["edges"], task["costs"], task["resources"])):
            lines.append(f"{(FAM[t] + f' ($s={task[chr(115)]}$, $t={task[chr(116)]}$, $R={task[chr(98)+chr(117)+chr(100)+chr(103)+chr(101)+chr(116)]}$)') if j == 0 else ''} & {j} & ({e[0]},{e[1]}) & {c_} & {r_} \\\\")
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{longtable}"]; (P / "tables/tableS6_instances.tex").write_text("\n".join(lines) + "\n")
    with open(P / "generated/numbers.tex", "w") as f:
        f.write("% AUTO-GENERATED by scripts/make_paper_assets.py from data/records. Do not edit.\n")
        words = "Zero One Two Three Four Five Six Seven Eight Nine".split()
        for k, v in N.items():
            name = "".join(words[int(c)] if c.isdigit() else c for c in k)
            f.write(f"\\newcommand{{\\num{name}}}{{{v}}}\n")
    (P / "generated/numbers.json").write_text(json.dumps(N, indent=1)); print(json.dumps(N, indent=1))

if __name__ == "__main__":
    main()
