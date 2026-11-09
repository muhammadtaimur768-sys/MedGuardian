#!/usr/bin/env bash
# Phase B (after E3 was checked on dev): E3 on test, E5 ablations + E2 on dev, E6 reliability seeds,
# then failure analysis, tables and figures. Safe to re-run: finished LLM calls come from the cache.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R="$PY -m src.medguardian.runner"
until [ -f results/e4_g3_test_summary.json ]; do sleep 30; done   # one LLM job at a time
$R --config experiments/configs/e3_medguardian.yaml --split test --run-name e3_medguardian_test50
for f in typed_tools calculator order_verifier; do
  $R --config experiments/configs/e5_no_$f.yaml --split dev --run-name e5_no_${f}_dev20
done
$R --config experiments/configs/e2_basepaper.yaml --split dev --run-name e2_basepaper_dev20
for s in 0 1 2; do
  $R --config experiments/configs/e6_t0.7_s$s.yaml --split dev --per-category 1 --run-name e6_t0.7_s${s}_dev10
done
$PY experiments/failure_analysis.py e3_medguardian_test50 e2_basepaper_test50
$PY experiments/make_tables.py
$PY experiments/make_figures.py
echo "PHASE B DONE"
