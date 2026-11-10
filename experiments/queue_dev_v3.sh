#!/usr/bin/env bash
# E3 on dev with all dev-failure fixes (reviewed before Phase B reaches E3 test), then E1.
set -euo pipefail
cd "$(dirname "$0")/.."
R=".venv/bin/python -m src.medguardian.runner"
$R --config experiments/configs/e3_medguardian.yaml --split dev --run-name e3_medguardian_dev20
echo "E3 DEV V3 DONE"
$R --config experiments/configs/e1_single_call.yaml --split test --run-name e1_single_call_test50
echo "DEV V3 DONE"
