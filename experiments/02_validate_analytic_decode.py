"""Step 2 - Internal validation: fast analytical decode-capacity model vs. the
iteration-level discrete-event simulator, on random (chip, TP, workload)
points.  Also fits the two calibration constants (GAMMA, KAPPA) on a training
half and reports error on the held-out half."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from regretsim.hardware import design_space, H100, H200, B200
from regretsim.model import LLAMA3_70B, fits
from regretsim.serving import decode_capacity_sim, decode_capacity_analytic
from regretsim.workloads import SCENARIOS

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
M = LLAMA3_70B


def pick_points(n=36, seed=1):
    rng = np.random.default_rng(seed)
    chips = design_space() + [H100, H200, B200]
    pts = []
    while len(pts) < n:
        c = chips[rng.integers(len(chips))]
        k = int(rng.choice([1, 2, 4, 8]))
        w = SCENARIOS[rng.integers(len(SCENARIOS))]
        if not fits(M, c, k):
            continue
        a, _ = decode_capacity_analytic(M, c, k, w, gamma=1.0, kappa=1.0)
        if a <= 0:
            continue
        pts.append((c, k, w))
    return pts


def run(p):
    c, k, w = p
    t = time.time()
    s, _ = decode_capacity_sim(M, c, k, w, n=2000)
    return dict(chip=c.name, k=k, wl=w.name, sim=s, secs=time.time() - t)


def main():
    pts = pick_points()
    with Pool(2) as pool:
        rows = pool.map(run, pts)
    grid = [(g, kp) for g in np.arange(1.0, 1.41, 0.05) for kp in np.arange(0.80, 1.001, 0.025)]
    idx = np.arange(len(pts))
    train, test = idx[::2], idx[1::2]

    def err(g, kp, ii):
        e = []
        for i in ii:
            c, k, w = pts[i]
            a, _ = decode_capacity_analytic(M, c, k, w, gamma=g, kappa=kp)
            s = rows[i]["sim"]
            if s > 0:
                e.append(abs(a - s) / s)
        return float(np.mean(e))
    g, kp = min(grid, key=lambda x: err(x[0], x[1], train))
    for i, (c, k, w) in enumerate(pts):
        rows[i]["analytic"] = decode_capacity_analytic(M, c, k, w, gamma=g, kappa=kp)[0]
    res = dict(gamma=round(float(g), 3), kappa=round(float(kp), 3),
               mape_train=err(g, kp, train), mape_test=err(g, kp, test),
               mape_uncalibrated=err(1.0, 1.0, idx), rows=rows)
    with open(os.path.join(OUT, "02_validation_analytic_decode.json"), "w") as f:
        json.dump(res, f, indent=1)
    print({k_: v for k_, v in res.items() if k_ != "rows"})


if __name__ == "__main__":
    main()
