"""Step 6 - All paper figures (PNG, 200 dpi) into figures/."""
import json, os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from regretsim.regret import tables
from regretsim.workloads import SCENARIOS, Workload
from regretsim.hardware import design_space, H100, H200, B200

ROOT = os.path.join(os.path.dirname(__file__), "..")
R, F = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(F, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MEMC = {"HBM3e": C[0], "GDDR7": C[1], "LPDDR5X": C[2]}
MEMM = {"HBM3e": "o", "GDDR7": "s", "LPDDR5X": "^"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
CH = {c.name: c for c in design_space() + [H100, H200, B200]}
SHORT = {"Code-2023": "Code\n2023", "Conv-2023": "Conv\n2023", "Reasoning (out x8)": "Reasoning\nout×8",
         "Agentic/RAG (ctx x16)": "Agentic\nctx×16", "Repo-scale code (ctx x16)": "Repo code\nctx×16",
         "Long-ctx reasoning (ctx x8, out x8)": "Long reas.\nctx×8 out×8"}


def save(fig, name):
    fig.savefig(os.path.join(F, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig1_workloads():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for i, b in enumerate(["code", "conv"]):
        c, o = Workload(b).lengths()
        for j, (x, lab) in enumerate([(c, "prompt"), (o, "output")]):
            xs = np.sort(x)
            ax[j].plot(xs, np.linspace(0, 1, len(xs)), color=C[i], lw=2, label=f"Azure {b}")
    for j, t in enumerate(["Prompt tokens", "Output tokens"]):
        ax[j].set_xscale("log")
        ax[j].set_xlabel(t)
        ax[j].set_ylabel("CDF")
    ax[0].legend(frameon=False)
    save(fig, "fig1_trace_cdfs.png")


def fig2_validation():
    v = json.load(open(os.path.join(R, "01_validation_mlperf.json")))
    names = list(v["chips"])
    meas = [v["chips"][n]["measured"] / 1e3 for n in names]
    pred = [v["chips"][n]["predicted"] / 1e3 for n in names]
    lo = [v["chips"][n]["band"][0] / 1e3 for n in names]
    hi = [v["chips"][n]["band"][1] / 1e3 for n in names]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, meas, 0.38, color=C[0], label="MLPerf measured")
    ax.bar(x + 0.2, pred, 0.38, color=C[1], label="Model", yerr=[np.array(pred) - lo, np.array(hi) - pred],
           ecolor=INK2, capsize=3)
    for i, n in enumerate(names):
        e = v["chips"][n]["error"] * 100
        ax.text(x[i] + 0.2, hi[i] + 0.8, f"{e:+.0f}%" if abs(e) > 0.5 else "calib.", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, [n.replace("-like", "") for n in names])
    ax.set_ylabel("8-accelerator Llama-2-70B\noffline throughput (k tok/s)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig2_validation_mlperf.png")


def fig3_dse():
    df = pd.read_csv(os.path.join(R, "03_dse.csv"))
    Cp, Cd, _ = tables(df)
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.9), sharey=False)
    panels = [(Cp, "Conv-2023", "Prefill chips · Conv-2023"), (Cd, "Conv-2023", "Decode chips · Conv-2023"),
              (Cd, "Reasoning (out x8)", "Decode chips · Reasoning (out×8)")]
    for a, (T, w, title) in zip(ax, panels):
        best = T[w].min()
        for m in MEMC:
            names = [n for n in T.index if n in CH and CH[n].mem == m and n not in ("H100-like", "H200-like", "B200-like")]
            ys = T.loc[names, w] / best
            ok = np.isfinite(ys)
            a.scatter([CH[n].tflops for n in np.array(names)[ok]], ys[ok], s=18, marker=MEMM[m], color=MEMC[m],
                      alpha=0.85, label=m, edgecolor=SURF, linewidth=0.5)
        for r, mk in (("H100-like", "X"), ("B200-like", "P")):
            a.scatter([CH[r].tflops], [T.loc[r, w] / best], marker=mk, s=60, color=INK, label=r.replace("-like", ""))
        a.set_yscale("log")
        a.set_ylim(0.9, 12)
        a.set_title(title, fontsize=9, color=INK)
        a.set_xlabel("Dense FP8 TFLOP/s per chip")
    ax[0].set_ylabel("Cost/request ÷ best")
    ax[2].legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "fig3_dse_scatter.png")


def fig4_regret():
    t = pd.read_csv(os.path.join(R, "04_regret_table.csv"), index_col=0)
    wls = [w.name for w in SCENARIOS]
    order = ["GPU-H100", "GPU-H200", "GPU-B200", "Unified-today", "Robust-unified", "Specialized-today", "Robust-specialized"]
    M = t.loc[order, wls].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
    im = ax.imshow(np.minimum(M, 150), cmap=cmap, vmin=0, vmax=150, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 80 else INK)
    ax.set_xticks(range(len(wls)), [SHORT[w] for w in wls], fontsize=7.5)
    ax.set_yticks(range(len(order)), order)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("Regret (extra cost vs. optimum)", fontsize=8)
    save(fig, "fig4_regret_matrix.png")


def fig5_drift():
    p = os.path.join(R, "05_drift_regret.csv")
    if not os.path.exists(p):
        print("skip fig5 (drift not done)")
        return
    d = pd.read_csv(p)
    strs = ["GPU-B200", "Unified-today", "Specialized-today"]
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 2.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
    for a, s in zip(ax, strs):
        g = d[d.strategy == s].pivot_table(index="ctx_scale", columns="out_scale", values="regret") * 100
        im = a.imshow(np.minimum(g.values, 150), cmap=cmap, vmin=0, vmax=150, origin="lower")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                a.text(j, i, f"{g.values[i,j]:.0f}", ha="center", va="center", fontsize=7.5,
                       color="white" if g.values[i, j] > 80 else INK)
        a.set_xticks(range(g.shape[1]), [f"×{x:g}" for x in g.columns])
        a.set_yticks(range(g.shape[0]), [f"×{x:g}" for x in g.index])
        a.set_xlabel("Output-length scaling")
        a.set_title(s, fontsize=9)
        a.grid(False)
    ax[0].set_ylabel("Context-length scaling")
    cb = fig.colorbar(im, ax=ax, fraction=0.02)
    cb.set_label("Regret (%)", fontsize=8)
    save(fig, "fig5_drift_heatmap.png")


def fig6_sensitivity():
    files = sorted(glob.glob(os.path.join(R, "05_sens_*.json")))
    if not files:
        print("skip fig6")
        return
    strs = ["GPU-B200", "Unified-today", "Robust-unified", "Specialized-today"]
    names, vals = ["baseline"], []
    t = pd.read_csv(os.path.join(R, "04_regret_table.csv"), index_col=0)
    vals.append([t.loc[s, "max"] * 100 for s in strs])
    for f in files:
        j = json.load(open(f))
        names.append(j["variant"])
        vals.append([j["reoptimized"][s]["max"] * 100 for s in strs])
    V = np.array(vals)
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    x = np.arange(len(names))
    w = 0.2
    for i, s in enumerate(strs):
        ax.bar(x + (i - 1.5) * w, V[:, i], w * 0.92, color=C[i], label=s)
    ax.set_xticks(x, names, fontsize=8, rotation=20)
    ax.set_ylabel("Worst-case regret (%)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_sensitivity.png")


def fig7_costsplit():
    b = pd.read_csv(os.path.join(R, "04_best_designs.csv"))
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    y = np.arange(len(b))
    ax.barh(y, 1 - b.decode_share, color=C[0], label="Prefill", height=0.6)
    ax.barh(y, b.decode_share, left=1 - b.decode_share + 0.004, color=C[1], label="Decode", height=0.6)
    for i, s in enumerate(b.decode_share):
        ax.text(1.02, i, f"{100*s:.0f}% decode", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [SHORT[w].replace("\n", " ") for w in b.workload])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Share of optimal serving cost")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_cost_split.png")


if __name__ == "__main__":
    fig1_workloads(); fig2_validation(); fig3_dse(); fig4_regret(); fig7_costsplit(); fig5_drift(); fig6_sensitivity()
