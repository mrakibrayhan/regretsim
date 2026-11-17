"""Step 4 - Specialization-regret analysis on the DSE results.
Outputs:
  results/04_best_designs.csv      optimal prefill / decode chip per workload
  results/04_regret_table.csv      regret of each design strategy on each workload
  results/04_candidates.txt        near-optimal chips used for the drift heat-map (step 5)
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from regretsim.regret import tables, choose, regret_table
from regretsim.workloads import TODAY, SCENARIOS
from regretsim.hardware import design_space, H100, H200, B200

R = os.path.join(os.path.dirname(__file__), "..", "results")
CH = {c.name: c for c in design_space() + [H100, H200, B200]}


def describe(name):
    c = CH[name]
    return dict(tflops=c.tflops, mem=c.mem, bw_TBs=round(c.bw / 1e12, 2), cap_GB=round(c.cap / 1e9),
                link=c.link, bom=round(c.bom()), tdp=round(c.tdp))


def main(dse=os.path.join(R, "03_dse.csv"), tag="04"):
    df = pd.read_csv(dse)
    Cp, Cd, wls = tables(df)
    order = [w.name for w in SCENARIOS if w.name in wls]
    today = [w.name for w in TODAY]
    # best designs per workload
    rows = []
    for w in order:
        p = Cp[w].idxmin()
        d = Cd[w].idxmin()
        kp = df[(df.chip == p) & (df.phase == "prefill") & (df.base == df[df.wl == w].base.iloc[0])
                & (df.ctx_scale == df[df.wl == w].ctx_scale.iloc[0])].k.iloc[0]
        kd = df[(df.chip == d) & (df.phase == "decode") & (df.wl == w)].k.iloc[0]
        rows.append(dict(workload=w, best_prefill=p, tp_prefill=kp, best_decode=d, tp_decode=kd,
                         cost_prefill_per_1k=Cp.loc[p, w] * 1e3, cost_decode_per_1k=Cd.loc[d, w] * 1e3,
                         decode_share=Cd.loc[d, w] / (Cp.loc[p, w] + Cd.loc[d, w]),
                         **{f"P_{k}": v for k, v in describe(p).items()},
                         **{f"D_{k}": v for k, v in describe(d).items()}))
    best = pd.DataFrame(rows)
    best.to_csv(os.path.join(R, f"{tag}_best_designs.csv"), index=False)
    strat = choose(Cp, Cd, today, order)
    rt = regret_table(Cp, Cd, strat, order)
    rt.to_csv(os.path.join(R, f"{tag}_regret_table.csv"))
    # candidate set: top-6 per phase per workload
    cand = set()
    for w in order:
        cand |= set(Cp[w].nsmallest(6).index) | set(Cd[w].nsmallest(6).index)
    for p, d in strat.values():
        cand |= {p, d}
    with open(os.path.join(R, f"{tag}_candidates.txt"), "w") as f:
        f.write("\n".join(sorted(cand)))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(best[["workload", "best_prefill", "tp_prefill", "best_decode", "tp_decode",
                "cost_prefill_per_1k", "cost_decode_per_1k", "decode_share"]].round(4).to_string())
    print()
    show = rt.copy()
    for c in order + ["max", "mean"]:
        show[c] = (100 * show[c]).round(1)
    print(show.to_string())
    print("candidates:", len(cand))
    return best, rt, strat


if __name__ == "__main__":
    main(*sys.argv[1:])
