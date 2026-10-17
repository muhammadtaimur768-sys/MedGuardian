# Upstream MedAgentBench baseline results

Unmodified MedAgentBench HTTPAgent (no MedGuardian tools) on a stratified subset: first 3 tasks of
each of the 10 task types = 30 tasks (`eval/run_baseline.py setup`). Graded by upstream refsol.py.

Paper reference: Claude 3.5 Sonnet v2 (Jiang et al., NEJM AI 2025): query 85.33%, action 54.00%, overall 69.67% (300 tasks). Our numbers are on 30 tasks with a different model, so they are
not directly comparable; they are an initial baseline only.

## Status: pending: not yet run

No `overall.json` found under `medagentbench/outputs/MedAgentBenchv1/`. Steps:

```bash
python3 eval/run_baseline.py setup     # subset data + overlay configs + generated Gemini agent yaml
docker run -p 8080:8080 jyxsu6/medagentbench:latest
python3 eval/run_baseline.py check     # FHIR server + refsol.py present
python3 eval/run_baseline.py start     # controller + 1 worker (leave running)
cd medagentbench && python -m src.assigner --config configs/assignments/medguardian_gemini.yaml
python3 eval/summarize_baseline.py
```
