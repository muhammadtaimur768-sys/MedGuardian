#!/usr/bin/env bash
# Wait for the E2 test run, then run E3 on dev (debugging) and E1 on test. One model at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
until [ -f results/e2_basepaper_test50_summary.json ]; do sleep 30; done
PY=.venv/bin/python
$PY -m src.medguardian.runner --config experiments/configs/e3_medguardian.yaml --split dev --run-name e3_medguardian_dev20
$PY -m src.medguardian.runner --config experiments/configs/e1_single_call.yaml --split test --run-name e1_single_call_test50
