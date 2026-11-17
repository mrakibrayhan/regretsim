# RegretSim: Specialization Regret in Phase-Disaggregated LLM Inference Hardware

Trace-driven simulation study. How much does a prefill/decode-specialized
accelerator fleet lose when LLM workloads drift (reasoning-length outputs,
agentic-length prompts), and how can hardware be designed to hedge against that?

Author: Md. Rakib Rayhan (North South University)

---

## Step-by-step guide (do these in order)

### Step 0: Set up (about 2 min)
```bash
python3 -m pip install numpy scipy pandas matplotlib
```
The Azure LLM Inference Trace 2023 CSVs are already in `data/`. Their source is
https://github.com/Azure/AzurePublicDataset (files `AzureLLMInferenceTrace_code.csv`
and `AzureLLMInferenceTrace_conv.csv`).

### Step 1: External validation against MLPerf (about 5 s)
```bash
python3 experiments/01_validate_mlperf.py
```
This calibrates the single free parameter (GEMM efficiency, η = 0.45) on the H100
MLPerf result and predicts the held-out H200 and MI300X results.
Output: `results/01_validation_mlperf.json`

### Step 2: Simulator vs. closed-form decode model (about 1 min)
```bash
python3 experiments/02_validate_analytic_decode.py
```
Shows why a Poisson-arrival analytical model is not enough for bursty production
traces, which is why every result uses the discrete-event simulator.
Output: `results/02_validation_analytic_decode.json`

### Step 3: Full design-space exploration (about 12 min on 2 cores)
```bash
python3 experiments/03_dse.py
```
Covers 144 chip designs plus H100/H200/B200 references, 6 workload futures, and
both phases. The run resumes automatically if interrupted.
Output: `results/03_dse.csv`

### Step 4: Regret analysis (about 5 s)
```bash
python3 experiments/04_regret.py
```
Outputs: `results/04_best_designs.csv`, `results/04_regret_table.csv`,
`results/04_candidates.txt`

### Step 5: Sensitivity and drift heat-map (about 25 min)
```bash
python3 experiments/05_drift_and_sensitivity.py prices   # memory-price variants (seconds)
python3 experiments/05_drift_and_sensitivity.py sens     # eta, TPOT, TTFT, fleet-size variants
python3 experiments/05_drift_and_sensitivity.py drift    # 5x5 context x output drift grid
```
Outputs: `results/05_*.json`, `results/05_drift_regret.csv`

### Step 6: Power-cap variant: 2.5 kW cap, adds 7500/9000 TFLOP/s designs (about 4 min)
```bash
python3 experiments/07_powercap.py
```
Output: `results/07_cap2500_*.csv`, `results/07_cap2500_summary.json`

### Step 7: Variance over 3 trace windows (about 4 min)
```bash
python3 experiments/08_variance.py
```
Output: `results/08_var_offset*.csv`, `results/08_variance_summary.json`

### Step 8: Figures (about 10 s)
```bash
python3 experiments/06_figures.py
```
Output: `figures/fig1..fig7*.png`

---

## Code map

| File | What it is |
|---|---|
| `regretsim/hardware.py` | Chip design space, memory-tech catalog, die/package/memory cost, power, TCO |
| `regretsim/model.py` | Llama shapes and the operator-level roofline time/energy model (TP = k) |
| `regretsim/workloads.py` | Azure traces, drift scaling, load-balanced (round-robin) arrival thinning |
| `regretsim/serving.py` | Prefill and continuous-batching decode discrete-event simulators, SLO capacity search, cost per request |
| `regretsim/regret.py` | Regret definition and design strategies (today-optimal, minimax-regret) |
| `experiments/0X_*.py` | One script per step above |

## Key assumptions to question first
All constants sit at the top of `hardware.py`, `model.py` and `serving.py`:
memory prices, energy per bit, η, SLOs (P90 TTFT ≤ 3× H100 unloaded, P90 TPOT ≤ 50 ms),
fleet rate (1000 req/s), and KV reservation (ctx + out).
## Data and license
- `data/` contains the Azure LLM Inference Trace 2023 from Microsoft's
  [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset); see that repository for its license and citation.
- Code: MIT License (see `LICENSE`).
