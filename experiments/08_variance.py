"""Step 8 - Variance over trace windows.  Re-runs the near-optimal candidate
designs on two additional windows of the (cyclic) Azure trace stream and
reports, per strategy, the worst-case regret in each window.  Window 0 reuses
the main sweep (03_dse.csv) restricted to the same candidates.
Outputs: results/08_var_offset<k>.csv, results/08_variance_summary.json"""
import json, os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
import regretsim.serving as S
import regretsim.hardware as H
from regretsim.workloads import SCENARIOS, TODAY
from regretsim.regret import tables, choose, regret_table, regret_of

R = os.path.join(os.path.dirname(__file__), "..", "results")
OFFSETS = (5000, 11000)
CH = {c.name: c for c in H.design_space() + [H.H100, H.H200, H.B200]}


def _init(off):
    S.SAMPLE_OFFSET = off


def _job(a):
    cname, wl, phase = a
    cost, k, cap = S.best_phase_cost(CH[cname], wl, phase)
    return dict(chip=cname, wl=wl.name, base=wl.base, ctx_scale=wl.ctx_scale, out_scale=wl.out_scale,
                phase=phase, cost=cost, k=k, cap=cap)


def main():
    cands = [l.strip() for l in open(os.path.join(R, "04_candidates.txt")) if l.strip()]
    order = [w.name for w in SCENARIOS]
    today = [w.name for w in TODAY]
    main_rt = pd.read_csv(os.path.join(R, "04_regret_table.csv"), index_col=0)
    frames = {0: pd.read_csv(os.path.join(R, "03_dse.csv"))}
    frames[0] = frames[0][frames[0].chip.isin(cands)]
    for off in OFFSETS:
        out = os.path.join(R, f"08_var_offset{off}.csv")
        if not os.path.exists(out):
            js = import_module("03_dse").jobs_for(SCENARIOS, cands)
            t = time.time()
            with Pool(2, initializer=_init, initargs=(off,)) as p:
                rows = p.map(_job, js, chunksize=2)
            pd.DataFrame(rows).to_csv(out, index=False)
            print("offset", off, len(rows), f"{time.time()-t:.0f}s", flush=True)
        frames[off] = pd.read_csv(out)
    res = {"offsets": [0] + list(OFFSETS), "n_candidates": len(cands), "fixed_designs": {}, "reoptimized": {}}
    for off, df in frames.items():
        Cp, Cd, _ = tables(df)
        for s in main_rt.index:  # designs fixed from the main study
            r = regret_of(Cp, Cd, main_rt.loc[s, "prefill_chip"], main_rt.loc[s, "decode_chip"], order)
            res["fixed_designs"].setdefault(s, []).append(float(r.max()))
        rt = regret_table(Cp, Cd, choose(Cp, Cd, today, order), order)
        for s in rt.index:
            res["reoptimized"].setdefault(s, []).append(
                dict(max=float(rt.loc[s, "max"]), p=rt.loc[s, "prefill_chip"], d=rt.loc[s, "decode_chip"]))
    json.dump(res, open(os.path.join(R, "08_variance_summary.json"), "w"), indent=1)
    for s, v in res["fixed_designs"].items():
        print(f"{s:20s} fixed-design max regret per window: {[round(100*x,1) for x in v]}")
    for s, v in res["reoptimized"].items():
        print(f"{s:20s} reoptimized: {[(round(100*x['max'],1), x['p'], x['d']) for x in v]}")


if __name__ == "__main__":
    main()
