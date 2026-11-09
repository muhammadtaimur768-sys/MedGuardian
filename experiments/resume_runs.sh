#!/usr/bin/env bash
# Resume the overnight queue after an interruption. Finished LLM calls replay from .cache/ instantly,
# so re-running a partly finished experiment only does the remaining work.
# Order: E2 (finish) -> E3 dev (reviewed before Phase B) -> E1 -> E4-G2 -> E4-G3.
set -euo pipefail
cd "$(dirname "$0")/.."
until curl -sf -o /dev/null localhost:8080/fhir/metadata; do sleep 5; done   # FHIR server back up
until curl -sf -o /dev/null localhost:11434/api/version; do sleep 2; done    # Ollama back up
PY=.venv/bin/python
R="$PY -m src.medguardian.runner"
$R --config experiments/configs/e2_basepaper.yaml --split test --run-name e2_basepaper_test50
$R --config experiments/configs/e3_medguardian.yaml --split dev --run-name e3_medguardian_dev20
$R --config experiments/configs/e1_single_call.yaml --split test --run-name e1_single_call_test50
$PY experiments/run_e4_g2.py --split test
$PY experiments/run_e4_g3.py --split test
echo "RESUME QUEUE DONE"
