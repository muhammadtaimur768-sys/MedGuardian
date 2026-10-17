# Rule-based (non-LLM) baselines on seed cases

Run: `python eval/run_rule_baseline.py` (mock FHIR server, no API key).

- **B0 no renal check** (requested dose given unchanged): 4/10 correct (40%)
- **B1 scripted tool chain** (FHIR -> Cockcroft-Gault -> dose table, fixed order): 10/10 correct (100%)

Ground truth is hand-computed (arithmetic in each case's `notes` in `eval/data/seed_cases.json`).
B1 shares code with the tools under test, so it checks the tools against the hand-computed values; it is not an independent agent result.

| Case | Drug | Requested mg | Expected CrCl / dose | B0 | B1 CrCl / dose | B1 |
|---|---|---|---|---|---|---|
| SC01 | metformin | 2000 | 39.0 / 1000 | FAIL | 39.0 / 1000.0 | PASS |
| SC02 | enoxaparin | 40 | 25.0 / 30 | FAIL | 25.0 / 30.0 | PASS |
| SC03 | gabapentin | 2400 | 105.0 / 2400 | PASS | 105.0 / 2400.0 | PASS |
| SC04 | gabapentin | 1800 | 29.2 / 700 | FAIL | 29.2 / 700.0 | PASS |
| SC05 | levetiracetam | 1000 | 82.5 / 1000 | PASS | 82.5 / 1000.0 | PASS |
| SC06 | levetiracetam | 1000 | 36.6 / 750 | FAIL | 36.6 / 750.0 | PASS |
| SC07 | dabigatran | 150 | 31.7 / 150 | PASS | 31.7 / 150.0 | PASS |
| SC08 | dabigatran | 150 | 18.0 / 75 | FAIL | 18.0 / 75.0 | PASS |
| SC09 | metformin | 2000 | 29.6 / 0 | FAIL | 29.6 / 0.0 | PASS |
| SC10 | enoxaparin | 40 | 82.6 / 40 | PASS | 82.6 / 40.0 | PASS |
