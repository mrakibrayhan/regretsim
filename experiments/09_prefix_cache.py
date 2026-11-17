"""Step 9 - Prefix-caching sensitivity for the long-context futures.
Agentic and coding-agent traffic reuses long prompt prefixes (prefix-cache hit
rates around 95% are reported for coding agents), so a fully-prefilled 16x
prompt may overstate prefill pressure.  This step re-simulates PREFILL for all
designs on the three long-context futures (Agent, Repo, LCR) with a prefix-hit
fraction h in {0.5, 0.9} (cached tokens are not recomputed but are attended to;
their KV is read from memory; full KV is still transferred to decode, which is
conservative).  Decode results and the short-context futures are reused from the
main sweep (they do not change).  Outputs: results/09_prefix_h*_prefill.csv,
results/09_prefix_summary.json"""
import json, os, sys, time
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import regretsim.serving as S
import regretsim.hardware as H
from regretsim.workloads import SCENARIOS, TODAY, Workload
from regretsim.regret import tables, choose, regret_table, regret_of

R = os.path.join(os.path.dirname(__file__), "..", "results")
HITS = (0.5, 0.9)
CH = {c.name: c for c in H.design_space() + [H.H100, H.H200, H.B200]}
LONG = sorted({(w.base, w.ctx_scale) for w in SCENARIOS if w.ctx_scale > 1})


def _init(h):
    S.PREFIX_HIT = h


def _job(a):
    cname, base, cs = a
    wl = Workload(base, cs, 1.0)
    cost, k, cap = S.best_phase_cost(CH[cname], wl, "prefill")
    return dict(chip=cname, wl=wl.name, base=base, ctx_scale=cs, out_scale=1.0,
                phase="prefill", cost=cost, k=k, cap=cap)


def main():
    main_df = pd.read_csv(os.path.join(R, "03_dse.csv"))
    main_rt = pd.read_csv(os.path.join(R, "04_regret_table.csv"), index_col=0)
    order = [w.name for w in SCENARIOS]
    today = [w.name for w in TODAY]
    summ = {"hits": list(HITS), "long_context_prefill_sets": [f"{b} x{c:g}" for b, c in LONG], "variants": {}}
    for h in HITS:
        out = os.path.join(R, f"09_prefix_h{int(h*100)}_prefill.csv")
        if not os.path.exists(out):
            js = [(c, b, cs) for c in CH for (b, cs) in LONG]
            t = time.time()
            with Pool(2, initializer=_init, initargs=(h,)) as p:
                rows = p.map(_job, js, chunksize=2)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"h={h}: {len(rows)} prefill searches in {time.time()-t:.0f}s", flush=True)
        new = pd.read_csv(out)
        keep = main_df[~((main_df.phase == "prefill") & (main_df.ctx_scale > 1))]
        df = pd.concat([keep, new], ignore_index=True)
        Cp, Cd, _ = tables(df)
        rt = regret_table(Cp, Cd, choose(Cp, Cd, today, order), order)
        fixed = {s: float(regret_of(Cp, Cd, main_rt.loc[s, "prefill_chip"], main_rt.loc[s, "decode_chip"], order).max())
                 for s in main_rt.index}
        best = {w: dict(prefill=Cp[w].idxmin(), decode=Cd[w].idxmin(),
                        decode_share=float(Cd[w].min() / (Cp[w].min() + Cd[w].min()))) for w in order}
        summ["variants"][str(h)] = dict(
            reoptimized={s: dict(max=float(rt.loc[s, "max"]), mean=float(rt.loc[s, "mean"]),
                                 p=rt.loc[s, "prefill_chip"], d=rt.loc[s, "decode_chip"]) for s in rt.index},
            main_designs_fixed_max=fixed, best=best)
    json.dump(summ, open(os.path.join(R, "09_prefix_summary.json"), "w"), indent=1)
    for h, v in summ["variants"].items():
        print("hit", h)
        for s, x in v["reoptimized"].items():
            print(f"  {s:20s} reopt max {100*x['max']:6.1f}%  fixed-main max {100*v['main_designs_fixed_max'][s]:6.1f}%  ({x['p']} / {x['d']})")
        for w, b in v["best"].items():
            print(f"  {w[:24]:24s} P={b['prefill']:22s} D={b['decode']:22s} decode share {100*b['decode_share']:.0f}%")


if __name__ == "__main__":
    main()
