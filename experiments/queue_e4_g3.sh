#!/usr/bin/env bash
# Run E4-G3 on the test split once E4-G2 has finished (one LLM job at a time).
set -euo pipefail
cd "$(dirname "$0")/.."
until [ -f results/e4_g2_test_summary.json ] && grep -q '"guardian"' results/e4_g2_test_summary.json; do sleep 30; done
.venv/bin/python experiments/run_e4_g3.py --split test
