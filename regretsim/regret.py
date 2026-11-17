"""Specialization-regret analysis on top of the DSE results table.

Definitions (per workload w, cost = USD per request, both phases):
    C(p, d, w) = Cp(p, w) + Cd(d, w)              disaggregated fleet, prefill chip p, decode chip d
    C*(w)      = min_p Cp(p, w) + min_d Cd(d, w)   clairvoyant optimum for w
    R(p, d, w) = C(p, d, w) / C*(w) - 1            specialization regret

Design strategies compared:
    GPU-H100 / GPU-H200 / GPU-B200    homogeneous general-purpose pools
    Unified-today                     one co-designed chip for both phases, optimized for today's traces
    Specialized-today (SPAD-style)    separate prefill & decode chips optimized for today's traces
    Robust-unified / Robust-special.  minimax-regret designs over the whole scenario set
"""
import numpy as np
import pandas as pd

REFS = ["H100-like", "H200-like", "B200-like"]


def tables(df):
    """Return Cp, Cd as DataFrames [chip x workload] (USD/request)."""
    pre = df[df.phase == "prefill"].copy()
    dec = df[df.phase == "decode"].copy()
    wls = dec.drop_duplicates("wl")[["wl", "base", "ctx_scale", "out_scale"]]
    # prefill rows were computed with out_scale=1; broadcast to every workload sharing ctx
    rows = []
    for _, w in wls.iterrows():
        p = pre[(pre.base == w.base) & (pre.ctx_scale == w.ctx_scale)]
        for _, r in p.iterrows():
            rows.append(dict(chip=r.chip, wl=w.wl, cost=r.cost))
    Cp = pd.DataFrame(rows).pivot_table(index="chip", columns="wl", values="cost")
    Cd = dec.pivot_table(index="chip", columns="wl", values="cost")
    Cp = Cp.fillna(np.inf)       # infeasible (cannot meet SLO / does not fit) = infinite cost
    Cd = Cd.fillna(np.inf)
    common = Cp.index.intersection(Cd.index)
    return Cp.loc[common], Cd.loc[common], list(Cd.columns)


def regret_of(Cp, Cd, p, d, wls):
    best = Cp[wls].min() + Cd[wls].min()
    return (Cp.loc[p, wls] + Cd.loc[d, wls]) / best - 1.0


def choose(Cp, Cd, design_wls, eval_wls, candidates=None):
    """Pick designs with information about `design_wls` only."""
    cand = [c for c in (candidates or Cp.index) if c not in REFS]
    bp = Cp[design_wls].min()
    bd = Cd[design_wls].min()
    # normalized cost per phase on the design workloads (mean over workloads)
    star = bp + bd
    out = {}
    for r in REFS:
        if r in Cp.index:
            out[f"GPU-{r.split('-')[0]}"] = (r, r)
    # unified: same chip both phases
    uni = ((Cp.loc[cand, design_wls] + Cd.loc[cand, design_wls]) / star).mean(axis=1)
    u = uni.idxmin()
    out["Unified-today"] = (u, u)
    # specialized: phases separable given a fixed weighting -> iterate (coordinate descent, exact enough)
    best = (np.inf, None, None)
    pcs = Cp.loc[cand, design_wls]
    dcs = Cd.loc[cand, design_wls]
    for p in cand:
        tot = ((pcs.loc[p] + dcs) / star).mean(axis=1)
        d = tot.idxmin()
        if tot[d] < best[0]:
            best = (tot[d], p, d)
    out["Specialized-today"] = (best[1], best[2])
    # robust (minimax regret over eval_wls)
    bp_e = Cp[eval_wls].min()
    bd_e = Cd[eval_wls].min()
    star_e = bp_e + bd_e
    pce = Cp.loc[cand, eval_wls]
    dce = Cd.loc[cand, eval_wls]
    uni_r = ((pce + dce) / star_e).max(axis=1)
    ur = uni_r.idxmin()
    out["Robust-unified"] = (ur, ur)
    best = (np.inf, None, None)
    for p in cand:
        mx = ((pce.loc[p] + dce) / star_e).max(axis=1)
        d = mx.idxmin()
        if mx[d] < best[0]:
            best = (mx[d], p, d)
    out["Robust-specialized"] = (best[1], best[2])
    return out


def regret_table(Cp, Cd, strategies, wls):
    rows = {}
    for s, (p, d) in strategies.items():
        rows[s] = regret_of(Cp, Cd, p, d, wls)
    t = pd.DataFrame(rows).T
    t["max"] = t[wls].max(axis=1)
    t["mean"] = t[wls].mean(axis=1)
    t["prefill_chip"] = [strategies[s][0] for s in t.index]
    t["decode_chip"] = [strategies[s][1] for s in t.index]
    return t
