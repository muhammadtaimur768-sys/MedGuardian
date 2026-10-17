# MedGuardian

AI4015 Agentic AI, Project G3-07, Phase 1 (Deliverable C).

MedGuardian is a tool-grounded agentic co-pilot for medication dosage safety (Gap 1), with planned
extensions for peri-operative process auditing (Gap 2) and voice-driven sonography reporting (Gap 3).
It builds on [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) (Jiang et al.,
*NEJM AI* 2025, MIT licence), which is included as a git submodule. In Phase 1 the Gap 1 workflow runs
end to end: an LLM orchestrator looks up a patient on a FHIR R4 server, reads creatinine and weight,
computes Cockcroft-Gault creatinine clearance, applies a renal dose table, and holds the result for
clinician approval. Gaps 2 and 3 are interfaces only. Architecture: [`docs/architecture.md`](docs/architecture.md).

## Team

| Role | Name | Roll no. |
|---|---|---|
| Group head (Module 1 / Gap 1) | Syed Muhammad Mustafa | 23i-0661 |
| Group member (infra, orchestration, Module 2 / sonography) | Muhammad bin Taimur | 23i-0810 |
| Supervisor | Naeem Tareen | |

## Repository layout

```
.
├── medagentbench/              # upstream MedAgentBench (git submodule, not edited)
├── baseline/configs/           # our overlay configs for the upstream baseline (agent template, etc.)
├── src/
│   ├── medguardian/
│   │   ├── state.py            # PatientContext, ToolCallRecord, DosingRecommendation
│   │   ├── llm_client.py       # OpenAI-compatible chat client with tool calling (Gemini default)
│   │   ├── orchestrator.py     # LLM tool-calling loop + human approval gate
│   │   └── tools/
│   │       ├── fhir_tools.py   # query_patient, query_observation
│   │       ├── dosing.py       # cockcroft_gault_crcl, renal_adjusted_dose (5 drugs)
│   │       └── stubs.py        # guideline_rag (Gap 1, Phase 2), timeline_verifier (Gap 2)
│   └── sonography/stub.py      # Gap 3 data contract (stub)
├── eval/
│   ├── data/seed_cases.json    # 10 synthetic renal dosing cases, hand-computed ground truth
│   ├── mock_fhir_server.py     # Docker-free FHIR R4 mock (port 8081)
│   ├── run_rule_baseline.py    # non-LLM baselines B0/B1
│   ├── run_tool_augmented.py   # MedGuardian agent on the seed cases
│   ├── run_baseline.py         # sets up / checks the upstream baseline run
│   ├── summarize_baseline.py   # query vs action success rates from upstream outputs
│   └── *_results.md            # results
├── tests/                      # pytest: state, dosing, FHIR tools, orchestrator
├── docs/architecture.md        # diagrams, contracts, limitations
├── demo.py                     # end-to-end Gap 1 demo
├── requirements-medguardian.txt
└── .env.example
```

## Setup

Python 3.9 or newer. The code works with pydantic 1.10 (upstream pin) and pydantic 2.

```bash
git clone --recurse-submodules <repo-url>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-medguardian.txt
cp .env.example .env               # then set GEMINI_API_KEY in .env
```

LLM defaults (in `.env.example`): `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`,
`LLM_MODEL=gemini-3.5-flash-lite`. Any OpenAI-compatible endpoint works (Groq, OpenAI, Ollama) by changing
`LLM_BASE_URL`, `LLM_MODEL` and the key. Check current model IDs and free-tier limits with the provider
before running.

## Running

Tests (no network, no API key; the orchestrator test uses a scripted fake LLM):

```bash
pytest
```

Mock FHIR server (synthetic seed patients on `http://localhost:8081/fhir/`). The demo and eval scripts
start it themselves when needed; run it on its own if you want to query it by hand:

```bash
python eval/mock_fhir_server.py
```

End-to-end demo (name/DOB -> patient -> creatinine/weight -> CrCl -> renal dose -> approve? [y/N]):

```bash
python demo.py --no-llm --case SC01   # fixed tool chain, no API key needed
python demo.py --case SC04            # LLM orchestrator (needs GEMINI_API_KEY)
python demo.py --fhir live --case SC01  # against the MedAgentBench Docker server instead of the mock
```

Rule-based baselines and the tool-augmented agent on the 10 seed cases:

```bash
python eval/run_rule_baseline.py      # writes eval/rule_baseline_results.md
python eval/run_tool_augmented.py     # writes eval/tool_augmented_results.md (needs API key)
```

### Upstream MedAgentBench baseline

This runs the unmodified MedAgentBench agent on a stratified subset of its tasks. It needs Docker,
the grader file and an API key.

1. Start the FHIR server:
   ```bash
   docker pull jyxsu6/medagentbench:latest
   docker run -p 8080:8080 jyxsu6/medagentbench:latest
   ```
   Wait until `http://localhost:8080/fhir/metadata` responds.
2. Download `refsol.py` from the Box link in the upstream README
   (https://stanfordmedicine.box.com/s/fizv0unyjgkb1r3a83rfn5p3dc673uho) and place it at
   `medagentbench/src/server/tasks/medagentbench/refsol.py`. Do not commit it (it is in `.gitignore`).
3. Install the upstream requirements (`pip install -r medagentbench/requirements.txt`), then generate
   the overlay configs and subset data, and check the setup:
   ```bash
   python eval/run_baseline.py setup
   python eval/run_baseline.py check
   ```
4. Run the task server and the assigner using the commands printed by `python eval/run_baseline.py setup`
   (`python -m src.start_task -a --config ...` and `python -m src.assigner --config ...` from inside
   `medagentbench/`, with config files under `medagentbench/configs/`).
5. Summarise the results:
   ```bash
   python eval/summarize_baseline.py   # writes eval/baseline_results.md
   ```

## Results

Numbers, model IDs, dates and per-case traces are in the results files:

- [`eval/rule_baseline_results.md`](eval/rule_baseline_results.md): non-LLM baselines on the 10 seed cases.
  B0 (no renal check, requested dose given unchanged): 4/10. B1 (scripted tool chain): 10/10. B1 shares
  code with the tools, so it checks the tools against the hand-computed values; it is not an agent result.
- [`eval/tool_augmented_results.md`](eval/tool_augmented_results.md): MedGuardian LLM agent on the seed cases.
- [`eval/baseline_results.md`](eval/baseline_results.md): upstream MedAgentBench agent on a task subset,
  split into query vs action tasks (paper reference for Claude 3.5 Sonnet v2: query 85.33%, action 54.00%,
  overall 69.67%).

Seed-case ground truth is hand-computed; the arithmetic is in each case's `notes` field in
`eval/data/seed_cases.json`. It does not come from the LLM or from the tools under test.

## Implemented vs stubbed

| Component | Gap | Status |
|---|---|---|
| State schema and contracts (`state.py`) | all | Implemented |
| FHIR tools `query_patient`, `query_observation` | Gap 1 | Implemented, tested against the mock server |
| `cockcroft_gault_crcl`, `renal_adjusted_dose` (5 drugs) | Gap 1 | Implemented, unit-tested incl. band edges |
| Orchestrator + OpenAI-compatible LLM adapter | Gap 1 | Implemented |
| Human approval gate | Gap 1 | CLI placeholder; no EHR writes |
| Mock FHIR server, seed dataset, runners | Gap 1 | Implemented |
| `guideline_rag` | Gap 1 (Phase 2) | Stub (returns "not implemented") |
| `timeline_verifier` | Gap 2 | Stub (returns "not implemented") |
| Sonography reporting (`src/sonography/stub.py`) | Gap 3 | Stub (data contract only) |

Known limitations are listed in [`docs/architecture.md`](docs/architecture.md#limitations).

## Handout §6 checklist

| Item | Where |
|---|---|
| Finalized architecture | `docs/architecture.md` |
| Working data/environment pipeline | `eval/mock_fhir_server.py`, `eval/data/seed_cases.json`, MedAgentBench submodule + `eval/run_baseline.py` |
| Baseline with initial results | `eval/rule_baseline_results.md`, `eval/baseline_results.md` |
| Model/provider setup | `src/medguardian/llm_client.py`, `.env.example`, `baseline/configs/` |
| Tool interfaces | `src/medguardian/tools/`, `src/sonography/stub.py` |
| Defined state/context + structured contracts | `src/medguardian/state.py`, `docs/architecture.md` |
| Initial evaluation dataset / test suite | `eval/data/seed_cases.json`, `tests/` |
| At least one end-to-end workflow | `demo.py`, `eval/run_tool_augmented.py` |
| Clean repo + README + commits from both members | this README, git history |

## Data and ethics

Only MedAgentBench's public synthetic FHIR patients and our hand-written synthetic seed cases are used.
No real patient or hospital data is in this repository. This is a research prototype for a course
project. It is not a medical device and must not be used for clinical decisions.
