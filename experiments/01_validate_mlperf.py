"""Step 1 - External validation against MLPerf Inference v4.0 (Llama-2-70B, Offline).

Published anchors:
  * 8x H200 (1000 W): 31,712 tokens/s  (MLPerf v4.0, NVIDIA)
  * H200 @1000 W is +45% over H100 offline  -> 8x H100 ~ 21,870 tokens/s
The OpenOrca subset caps input and output at 1,024 tokens; exact means are not
published in the sources used, so we report a band over plausible means.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
from regretsim.hardware import H100, H200, Chip
from regretsim.model import LLAMA2_70B, prefill_batch, decode_iter, fits, kv_capacity_tokens

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def offline_tps_per_chip(chip, model, mean_in, mean_out, bmax=1024):
    best = 0
    for k in (1, 2, 4, 8):
        if not fits(model, chip, k):
            continue
        B = int(min(bmax, kv_capacity_tokens(model, chip, k) / (mean_in + mean_out)))
        if B < 1:
            continue
        # prefill B prompts in 16k-token chunks, then decode the batch to completion
        per = max(1, 16384 // mean_in)
        tp = 0.0
        left = B
        while left > 0:
            nb = min(per, left)
            tp += prefill_batch(model, chip, k, [mean_in] * nb)[0]
            left -= nb
        js = np.arange(mean_out)
        td = decode_iter(model, chip, k, B, B * (mean_in + js))[0].sum()
        tps = B * mean_out / (tp + td) / k
        best = max(best, tps)
    return best


MEASURED_8GPU = {  # MLPerf Inference Llama-2-70B Offline, tokens/s, 8 accelerators
    "H100-like": 24525.0,    # NVIDIA, v4.0/v4.1
    "H200-like": 31712.0,    # NVIDIA, v4.0 (1000 W custom thermal)
    "MI300X-like": 23514.0,  # AMD, v4.1
}
MI300X = Chip("MI300X-like", 2615, "HBM3", 8, link="fast", bw_override=5.3e12, cap_override=192e9)
MEAN_OUT = 294   # MLPerf target generation length (294.45 tokens)
MEAN_IN_ASSUMED = 250
IN_BAND = (150, 250, 350)


def predict(chip, eta, mi):
    return 8 * offline_tps_per_chip(replace(chip, eta_c=eta), LLAMA2_70B, mi, MEAN_OUT)


def main():
    # 1) calibrate the single free parameter (GEMM efficiency eta_c) on H100 only
    lo, hi = 0.2, 0.9
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if predict(H100, mid, MEAN_IN_ASSUMED) < MEASURED_8GPU["H100-like"]:
            lo = mid
        else:
            hi = mid
    eta = round(0.5 * (lo + hi), 3)
    res = dict(calibrated_eta_c=eta, mean_in_assumed=MEAN_IN_ASSUMED, mean_out=MEAN_OUT, chips={})
    # 2) held-out predictions (H200, MI300X) + band over unknown mean input length
    for chip in (H100, H200, MI300X):
        meas = MEASURED_8GPU[chip.name]
        pts = {mi: predict(chip, eta, mi) for mi in IN_BAND}
        res["chips"][chip.name] = dict(
            measured=meas, predicted=pts[MEAN_IN_ASSUMED],
            error=(pts[MEAN_IN_ASSUMED] - meas) / meas,
            band=[min(pts.values()), max(pts.values())],
            role="calibration" if chip is H100 else "held-out")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "01_validation_mlperf.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
