"""Step 7 - Power-cap sensitivity / design-range extension.
The main study's 6000 TFLOP/s optimum is the largest compute that fits the
1.8 kW package cap (every 7500 and 9000 TFLOP/s design exceeds it).  This step
admits 7500 and 9000 TFLOP/s tiers under a 2.5 kW cap, simulates only the new
designs, merges them with the main sweep, and re-runs the regret analysis.
Outputs: results/07_cap2500_dse_extra.csv, 07_cap2500_regret_table.csv,
         07_cap2500_best_designs.csv, 07_cap2500_summary.json"""
import json, os, sys, time
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
import regretsim.hardware as H
from regretsim.serving import best_phase_cost
from regretsim.workloads import SCENARIOS, TODAY
from regretsim.regret import tables, choose, regret_table

R = os.path.join(os.path.dirname(__file__), "..", "results")
CAP = 2500.0
EXTRA = {c.name: c for c in H.design_space(tfs=(7500, 9000), cap_w=CAP)}


def job(a):
    cname, wl, phase = a
    t = time.time()
    cost, k, cap = best_phase_cost(EXTRA[cname], wl, phase)
    return dict(chip=cname, wl=wl.name, base=wl.base, ctx_scale=wl.ctx_scale, out_scale=wl.out_scale,
                phase=phase, cost=cost, k=k, cap=cap, secs=time.time() - t)


def main():
    out = os.path.join(R, "07_cap2500_dse_extra.csv")
    if not os.path.exists(out):
        js = import_module("03_dse").jobs_for(SCENARIOS, list(EXTRA))
        print("jobs", len(js), "extra designs", len(EXTRA), flush=True)
        with Pool(2) as p:
            rows = p.map(job, js, chunksize=2)
        pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.concat([pd.read_csv(os.path.join(R, "03_dse.csv")), pd.read_csv(out)], ignore_index=True)
    Cp, Cd, wls = tables(df)
    order = [w.name for w in SCENARIOS]
    strat = choose(Cp, Cd, [w.name for w in TODAY], order)
    rt = regret_table(Cp, Cd, strat, order)
    rt.to_csv(os.path.join(R, "07_cap2500_regret_table.csv"))
    best = [dict(workload=w, best_prefill=Cp[w].idxmin(), best_decode=Cd[w].idxmin()) for w in order]
    pd.DataFrame(best).to_csv(os.path.join(R, "07_cap2500_best_designs.csv"), index=False)
    main_rt = pd.read_csv(os.path.join(R, "04_regret_table.csv"), index_col=0)
    # main-study designs evaluated against the widened optimum
    from regretsim.regret import regret_of
    keep = {s: float(regret_of(Cp, Cd, main_rt.loc[s, "prefill_chip"], main_rt.loc[s, "decode_chip"], order).max())
            for s in main_rt.index}
    summ = dict(cap_w=CAP, n_extra=len(EXTRA), n_total=int(len(Cp.index)),
                reoptimized={s: dict(max=float(rt.loc[s, "max"]), mean=float(rt.loc[s, "mean"]),
                                     p=rt.loc[s, "prefill_chip"], d=rt.loc[s, "decode_chip"]) for s in rt.index},
                main_designs_max_regret_vs_widened_optimum=keep,
                best=best)
    json.dump(summ, open(os.path.join(R, "07_cap2500_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
