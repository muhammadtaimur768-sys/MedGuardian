#!/usr/bin/env bash
# After the resume queue: E3 on dev with the dev-failure fixes (reviewed before test), then the new E1.
set -euo pipefail
cd "$(dirname "$0")/.."
until grep -q "RESUME QUEUE DONE" logs/resume_runs.log; do sleep 30; done
R=".venv/bin/python -m src.medguardian.runner"
$R --config experiments/configs/e3_medguardian.yaml --split dev --run-name e3_medguardian_dev20
$R --config experiments/configs/e1_single_call.yaml --split test --run-name e1_single_call_test50
echo "DEV V2 DONE"
