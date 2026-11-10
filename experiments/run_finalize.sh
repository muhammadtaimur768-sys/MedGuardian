#!/usr/bin/env bash
# After the Gap 2 LLM test arms: messy-records LLM arms, then regenerate every derived artefact.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until grep -q '^llm_rag {' logs/e4_g2_test_llm_v2.log 2>/dev/null; do sleep 30; done
echo "G2 TEST LLM DONE"
$PY experiments/run_e4_g2.py --split test_messy --arms llm_only llm_rag
$PY experiments/failure_analysis.py e3_medguardian_test50 e2_basepaper_test50
$PY experiments/export_samples.py
$PY experiments/make_tables.py
$PY experiments/make_figures.py
echo "FINALIZE DONE"
