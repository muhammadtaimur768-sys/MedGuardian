# Tool-augmented MedGuardian agent: seed-case results

- Model: `gemini-3.5-flash-lite` via `https://generativelanguage.googleapis.com/v1beta/openai` (OpenAI-compatible API), temperature 0
- Date: 2026-10-17 18:01 UTC
- FHIR: mock server (`eval/mock_fhir_server.py`), synthetic patients only
- Command: `python eval/run_tool_augmented.py`
- **Result: 10/10 cases correct (100%)** (CrCl within 0.05 mL/min and exact adjusted dose / contraindication)

| Case | Drug | Result | Agent output | Expected mg | Tool-call trace | Latency (s) | Tokens in/out |
|---|---|---|---|---|---|---|---|
| SC01 | metformin | PASS | CrCl 39.0, 1000.0 mg, contra=False | 1000 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 9.61 | 4546/320 (5 calls) |
| SC02 | enoxaparin | PASS | CrCl 25.0, 30.0 mg, contra=False | 30 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 24.02 | 4517/301 (5 calls) |
| SC03 | gabapentin | PASS | CrCl 105.0, 2400.0 mg, contra=False | 2400 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 9.96 | 4523/294 (5 calls) |
| SC04 | gabapentin | PASS | CrCl 29.2, 700.0 mg, contra=False | 700 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 9.05 | 4534/315 (5 calls) |
| SC05 | levetiracetam | PASS | CrCl 82.5, 1000.0 mg, contra=False | 1000 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 40.58 | 4557/335 (5 calls) |
| SC06 | levetiracetam | PASS | CrCl 36.6, 750.0 mg, contra=False | 750 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 7.53 | 4565/336 (5 calls) |
| SC07 | dabigatran | PASS | CrCl 31.7, 150.0 mg, contra=False | 150 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 24.77 | 4554/318 (5 calls) |
| SC08 | dabigatran | PASS | CrCl 18.0, 75.0 mg, contra=False | 75 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 8.85 | 4541/323 (5 calls) |
| SC09 | metformin | PASS | CrCl 29.6, 0.0 mg, contra=True | 0 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 10.27 | 4528/339 (5 calls) |
| SC10 | enoxaparin | PASS | CrCl 82.6, 40.0 mg, contra=False | 40 | query_patient → query_observation → query_observation → cockcroft_gault_crcl → renal_adjusted_dose | 7.64 | 4522/298 (5 calls) |

Full traces (every ToolCallRecord and the final PatientContext) are in `outputs/tool_augmented_runs.jsonl` (not committed; regenerate with the command above).
