#!/bin/sh
# Runs every step in order (about 40 min on 2 cores). See README.md.
set -e
python3 experiments/01_validate_mlperf.py
python3 experiments/02_validate_analytic_decode.py
python3 experiments/03_dse.py
python3 experiments/04_regret.py
python3 experiments/05_drift_and_sensitivity.py prices
python3 experiments/05_drift_and_sensitivity.py sens
python3 experiments/05_drift_and_sensitivity.py drift
python3 experiments/06_figures.py
