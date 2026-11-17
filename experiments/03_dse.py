"""Step 3 - Full design-space exploration.  For every chip in the design space
(plus H100/H200/B200-like references) and every workload scenario, find the
SLO-constrained capacity and cost/request of each phase using the
trace-driven discrete-event simulators.  Results cached to results/03_dse.csv."""
import os, sys, time
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from regretsim.hardware import design_space, H100, H200, B200
from regretsim.serving import best_phase_cost
from regretsim.workloads import SCENARIOS, Workload

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "03_dse.csv")
CHIPS = {c.name: c for c in design_space() + [H100, H200, B200]}


def job(args):
    cname, wl, phase = args
    c = CHIPS[cname]
    t = time.time()
    cost, k, cap = best_phase_cost(c, wl, phase)
    return dict(chip=cname, wl=wl.name, base=wl.base, ctx_scale=wl.ctx_scale,
                out_scale=wl.out_scale, phase=phase, cost=cost, k=k, cap=cap,
                secs=time.time() - t)


def jobs_for(workloads, chips):
    js = []
    seen_prefill = set()
    for wl in workloads:
        for cn in chips:
            js.append((cn, wl, "decode"))
            key = (cn, wl.base, wl.ctx_scale)   # prefill is independent of output length
            if key not in seen_prefill:
                seen_prefill.add(key)
                js.append((cn, Workload(wl.base, wl.ctx_scale, 1.0), "prefill"))
    return js


def run(js, out):
    done = set()
    rows = []
    if os.path.exists(out):
        old = pd.read_csv(out)
        rows = old.to_dict("records")
        done = set(zip(old.chip, old.base, old.ctx_scale, old.out_scale, old.phase))
    js = [j for j in js if (j[0], j[1].base, j[1].ctx_scale, j[1].out_scale, j[2]) not in done]
    print("jobs to run:", len(js), flush=True)
    t0 = time.time()
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(job, js, chunksize=2)):
            rows.append(r)
            if i % 50 == 0 or i == len(js) - 1:
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{i+1}/{len(js)}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    run(jobs_for(SCENARIOS, list(CHIPS)), OUT)
