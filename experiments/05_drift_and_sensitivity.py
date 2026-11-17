"""Step 5 - (a) Drift heat-map: regret of each strategy over a 5x5 grid of
context x output scaling of the Azure conversation trace.  (b) Sensitivity:
re-run the DSE (on the near-optimal candidate set) with perturbed model,
SLO, fleet and price assumptions and check whether the conclusions hold.

usage: python 05_drift_and_sensitivity.py drift | sens <variant> | prices
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from dataclasses import replace
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import regretsim.serving as S
import regretsim.workloads as W
import regretsim.hardware as H
from regretsim.workloads import Workload, SCENARIOS, TODAY
from regretsim.regret import tables, choose, regret_table, regret_of

R = os.path.join(os.path.dirname(__file__), "..", "results")
VARIANTS = {
    "eta075":   dict(eta=0.75),            # optimistic GEMM efficiency
    "tpot30":   dict(tpot=0.030),          # tight interactivity SLO
    "tpot100":  dict(tpot=0.100),          # relaxed SLO
    "ttft5":    dict(ttft=5.0),            # relaxed TTFT SLO
    "fleet200": dict(fleet=200.0),         # smaller fleet -> burstier per instance
}
_CHIPS = {}


def _init(ov):
    if "tpot" in ov:
        S.TPOT_SLO = ov["tpot"]
    if "ttft" in ov:
        S.TTFT_SLO_NORM = ov["ttft"]
    if "fleet" in ov:
        W.FLEET_RATE = ov["fleet"]
    base = {c.name: c for c in H.design_space() + [H.H100, H.H200, H.B200]}
    if "eta" in ov:
        base = {k: replace(c, eta_c=ov["eta"]) for k, c in base.items()}
    _CHIPS.update(base)


def _job(a):
    cname, wl, phase = a
    cost, k, cap = S.best_phase_cost(_CHIPS[cname], wl, phase)
    return dict(chip=cname, wl=wl.name, base=wl.base, ctx_scale=wl.ctx_scale,
                out_scale=wl.out_scale, phase=phase, cost=cost, k=k, cap=cap)


def run(chips, wls, ov, out):
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module
    jobs_for = import_module("03_dse").jobs_for
    js = jobs_for(wls, chips)
    t = time.time()
    with Pool(2, initializer=_init, initargs=(ov,)) as p:
        rows = p.map(_job, js, chunksize=2)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, len(rows), f"{time.time()-t:.0f}s", flush=True)


def candidates():
    with open(os.path.join(R, "04_candidates.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def strategies_from_main():
    df = pd.read_csv(os.path.join(R, "03_dse.csv"))
    Cp, Cd, wls = tables(df)
    order = [w.name for w in SCENARIOS]
    return choose(Cp, Cd, [w.name for w in TODAY], order)


def drift():
    grid = [Workload("conv", c, o) for c in (1, 2, 4, 8, 16) for o in (1, 2, 4, 8, 16)]
    out = os.path.join(R, "05_drift_dse.csv")
    if not os.path.exists(out):
        run(candidates(), grid, {}, out)
    df = pd.read_csv(out)
    Cp, Cd, wls = tables(df)
    strat = strategies_from_main()
    rows = []
    for s, (p, d) in strat.items():
        r = regret_of(Cp, Cd, p, d, wls)
        for w in wls:
            g = df[df.wl == w].iloc[0]
            rows.append(dict(strategy=s, ctx_scale=g.ctx_scale, out_scale=g.out_scale, regret=r[w]))
    pd.DataFrame(rows).to_csv(os.path.join(R, "05_drift_regret.csv"), index=False)
    t = pd.DataFrame(rows).groupby("strategy").regret.agg(["max", "mean"]) * 100
    print(t.round(1))


def sens(name):
    out = os.path.join(R, f"05_sens_{name}.csv")
    if not os.path.exists(out):
        run(candidates(), SCENARIOS, VARIANTS[name], out)
    summarize(out, name)


def summarize(path, name):
    df = pd.read_csv(path)
    Cp, Cd, wls = tables(df)
    order = [w.name for w in SCENARIOS]
    strat = choose(Cp, Cd, [w.name for w in TODAY], order)
    rt = regret_table(Cp, Cd, strat, order)
    # also: regret of the MAIN-run designs under this variant (are they still good?)
    main = strategies_from_main()
    keep = {}
    for s, (p, d) in main.items():
        if p in Cp.index and d in Cd.index:
            r = regret_of(Cp, Cd, p, d, order)
            keep[s] = dict(max=r.max(), mean=r.mean())
    res = dict(variant=name,
               reoptimized={s: dict(max=rt.loc[s, "max"], mean=rt.loc[s, "mean"],
                                    p=rt.loc[s, "prefill_chip"], d=rt.loc[s, "decode_chip"]) for s in rt.index},
               main_designs_under_variant=keep)
    with open(os.path.join(R, f"05_sens_{name}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(name, json.dumps({s: round(v["max"] * 100, 1) for s, v in res["reoptimized"].items()}))


def prices():
    """Cost-only perturbations: no re-simulation needed (capacity is price-independent)."""
    df = pd.read_csv(os.path.join(R, "03_dse.csv"))
    chips = {c.name: c for c in H.design_space() + [H.H100, H.H200, H.B200]}
    base_fixed = {n: c.fixed_usd_per_h() for n, c in chips.items()}
    cases = {"HBM10": ("HBM3e", 10.0), "HBM25": ("HBM3e", 25.0),
             "GDDR7_10": ("GDDR7", 10.0), "LPDDR_2": ("LPDDR5X", 2.0)}
    out = {}
    for name, (tech, price) in cases.items():
        old = H.MEMTECH[tech]["usd_gb"]
        H.MEMTECH[tech]["usd_gb"] = price
        new_fixed = {n: c.fixed_usd_per_h() for n, c in chips.items()}
        H.MEMTECH[tech]["usd_gb"] = old
        d2 = df.copy()
        ok = np.isfinite(d2.cost) & (d2.cap > 0)
        kk = d2.k.fillna(1)
        e_cost = d2.cost - kk * d2.chip.map(base_fixed) / (3600 * d2.cap)
        d2.loc[ok, "cost"] = (kk * d2.chip.map(new_fixed) / (3600 * d2.cap) + e_cost)[ok]
        p = os.path.join(R, f"05_price_{name.replace('/', '').replace('$', '')}.csv")
        d2.to_csv(p, index=False)
        summarize(p, name)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "drift":
        drift()
    elif cmd == "sens":
        for v in (sys.argv[2:] or VARIANTS):
            sens(v)
    elif cmd == "prices":
        prices()
