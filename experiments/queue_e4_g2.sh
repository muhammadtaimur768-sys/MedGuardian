#!/usr/bin/env bash
# Run E4-G2 (all three arms, test split) once the E1 run has finished. One model at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
until [ -f results/e1_single_call_test50_summary.json ]; do sleep 30; done
.venv/bin/python experiments/run_e4_g2.py --split test
