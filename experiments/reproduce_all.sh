#!/usr/bin/env bash
# Regenerate every result, table and figure from scratch (needs the FHIR server, refsol.py and the
# two Ollama models; see README). Cached LLM responses in .cache/ make re-runs instant; delete
# .cache/ to query the models again. Expect several hours on an Apple M4 with an empty cache.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R="$PY -m src.medguardian.runner"

$PY data/make_splits.py --seed 0
$PY -m pytest -q

$R --config experiments/configs/e0_mistral.yaml --split test --per-category 2 --run-name e0_mistral_test20
$R --config experiments/configs/e1_single_call.yaml --split test --run-name e1_single_call_test50
$R --config experiments/configs/e2_basepaper.yaml --split test --run-name e2_basepaper_test50
$R --config experiments/configs/e3_medguardian.yaml --split test --run-name e3_medguardian_test50

$PY experiments/run_e4_g2.py --split test
$PY experiments/run_e4_g3.py --split test

$R --config experiments/configs/e3_medguardian.yaml --split dev --run-name e3_medguardian_dev20
$R --config experiments/configs/e2_basepaper.yaml --split dev --run-name e2_basepaper_dev20
for f in typed_tools calculator order_verifier; do
  $R --config experiments/configs/e5_no_$f.yaml --split dev --run-name e5_no_${f}_dev20
done
for s in 0 1 2; do
  $R --config experiments/configs/e6_t0.7_s$s.yaml --split dev --per-category 1 --run-name e6_t0.7_s${s}_dev10
done

$PY experiments/run_e4_g2.py --split test_messy --arms guardian
$PY experiments/failure_analysis.py e3_medguardian_test50 e2_basepaper_test50
$PY experiments/export_samples.py
$PY experiments/make_tables.py
$PY experiments/make_figures.py
