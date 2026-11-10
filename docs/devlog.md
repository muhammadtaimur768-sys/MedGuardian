# Development log (Phase 2)

One entry per major iteration: the failure or limitation observed, the change made, and its
measured effect. Numbers come from files in `results/` or `logs/` unless stated.

---

### 2026-11-09 — Phase 1 evaluation was circular
- **Observed.** Phase 1's "10/10 (100 %)" was 10 hand-written cases served by a mock FHIR server
  generated from those same cases, and graded by comparing the deterministic tool's own output to
  a hand-computed value. `eval/baseline_results.md` said "pending"; the `medagentbench/` submodule
  was empty; no MedAgentBench number existed.
- **Change.** Evaluate on the real MedAgentBench Docker FHIR server (HAPI 7.6.0, 300 tasks) with the
  base paper's own grader (`refsol.py`), on fixed seed-0 splits (`data/splits/`). Gap 2/3 gold labels
  are injected by construction in generators that never call the system under test.
- **Effect.** The Phase 1 figure is withdrawn. First faithful numbers: E0 Mistral-7B v0.3 5.0 % overall
  on 20 test tasks vs 4.00 % reported by the base paper (`results/e0_mistral_test20_summary.json`).

### 2026-11-09 — Local inference 3–4x slower than the hardware allows
- **Observed.** Qwen2.5-7B on the M4 processed a 4.9k-token prompt at 38–48 tok/s and generated at
  ~5 tok/s. The Mac was on battery in Low Power Mode with 9.6 GB of 10 GB swap in use.
- **Change.** Mains power, Low Power Mode off, reboot, heavy apps closed; Docker capped at 6 GB; one
  Ollama slot; `caffeinate` around every run.
- **Effect.** 127 tok/s prompt processing and 15 tok/s generation on the same prompt (about 3x).

### 2026-11-10 — Ollama silently truncates over-long conversations
- **Observed.** A 29,774-token prompt returned HTTP 200 after Ollama cut it to 8,194 tokens (first 4
  tokens + tail); the model answered the mangled prompt. A faithful harness must instead report the
  base paper's "agent context limit" outcome.
- **Change.** `llm/pool.py` rejects a prompt before sending when a safe lower bound (chars / 6) already
  exceeds the window, and after the reply when the server reports fewer prompt tokens than the
  conversation must contain. Tested in `tests/test_pool_tracing_splits.py`.
- **Effect.** Overflow is now counted as `agent context limit` instead of being graded on a truncated
  prompt; the measured chars/token on FHIR JSON is 2.8, so the bound has a 2x margin.

### 2026-11-10 — Base-paper agent failure modes on dev (motivation for Gap 1 components)
- **Observed.** In the first 5 dev tasks with the unmodified agent (`results/smoke_e2_dev5.csv`):
  2/5 looped on invalid FHIR date-range syntax (HTTP 400) until the 8-round cap, 1/5 posted a vital
  sign to `/fhir/` instead of `/fhir/Observation`, 1/5 got the age wrong (64 vs 84). 1/5 correct.
- **Change.** Gap 1 components: typed FHIR tools (date-range rewrite, error hints, all pages,
  filtered newest-first summaries), calculator (age as of "now", hours-before-now, in-window
  statistics) and the pre-order verifier (endpoint, schema, patient, codes, indication, dose; one
  repair round).
- **Effect.** See the E3 vs E2 dev entry below.

### 2026-11-10 — URL normaliser decoded "+00:00" as a space
- **Observed.** Dry-running the verifier and typed tools on dev without the LLM, the date-range
  rewrite never fired: `parse_qsl` turned the `+` of `+00:00` into a space, so the timestamp did not
  parse and the URL went to the server unchanged (HTTP 400).
- **Change.** Escape `+` before parsing; re-encode it as `%2B`.
- **Effect.** Both malformed forms seen in the smoke run now return HTTP 200
  (`tests/test_gap1_components.py::test_normalise_rewrites_date_ranges_and_dedupes_format`).

### 2026-11-10 — Gap 3 ASR crashed on PyAV 16
- **Observed.** `faster-whisper` 1.2.1 passes `metadata_errors=` to `av.open`, removed in PyAV 16.
- **Change.** Read the 16 kHz mono WAV with the standard-library `wave` module and pass samples.
- **Effect.** ASR runs; mean WER 7.1 % on the 5 dev clips (`results/e4_g3_dev_summary.json`).

### 2026-11-10 — Base-paper agent with Qwen2.5-7B: near-zero on the test split
- **Observed.** E2 on test-50: 6.0 % overall, 12.0 % query, 0.0 % action; 28/50 tasks hit the 8-round
  limit, mostly by repeating the same malformed FHIR search after an HTTP 400
  (`results/e2_basepaper_test50_summary.json`).
- **Change.** None to E2 (it is the faithful baseline). This is the failure mode Gap 1 targets.
- **Effect.** Reference point for E3 (see below) and consistent with the base paper's small-model results
  (Mistral-7B 4.00 %, our E0 5.0 %).

### 2026-11-10 — E1 as a 1-round cap could not succeed by construction
- **Observed.** The first E1 (base-paper prompt, one round) scored 0/50 with every task at the round limit:
  any task needs either data (a GET) or a POST followed by FINISH, which one round cannot hold
  (`results/e1_onecap_test50_superseded_summary.json`).
- **Change.** E1 redefined as a non-agentic retrieve-then-answer baseline: a fixed pipeline fetches the
  patient and every lab code named in the task, then one call writes all POSTs and FINISH.
- **Effect.** E1 now measures what the agentic loop adds over a fixed pipeline with the same data
  (result in `results/e1_single_call_test50_summary.json`).

### 2026-11-10 — E3 on dev, round 1: search mistakes, not reasoning
- **Observed.** E3 on dev-20 scored 60.0 % (12/20; `results/e3_medguardian_dev20_v1_summary.json`).
  All 8 failures were FHIR search problems: full-name Patient search (HAPI returns 0), model-chosen date
  windows (wrong day, or exactly "now"), `_sort=-effectiveDateTime` (HTTP 400), searching by the ordering
  LOINC code instead of the lab code, and a GET with a POST glued onto it.
- **Change.** Typed tools split full names into given/family, drop model-chosen date filters when the
  calculator is on (it computes the task's window), drop `_sort`/`_count`, hint the task's lab code on an
  empty search, and execute only the first GET of a multi-action reply.
- **Effect.** 60.0 % → **75.0 %** on dev-20 (query 60 → 90 %, action 60 → 60 %;
  `results/e3_medguardian_dev20_v2_summary.json`).

### 2026-11-10 — E3 on dev, round 2: the model ignores hints; one result list overflows
- **Observed.** In round-1-fixed runs the model still searched potassium/HbA1c by the ordering LOINC
  code, received the hint, and answered "not found" anyway (4 tasks); an average-glucose task returned
  203 readings and overflowed the 16k context.
- **Change.** Deterministic query repair: an empty Observation search by a code other than the one the
  task names is re-run with the named lab code (reported to the model). Results are capped at the
  newest 30 resources; the calculator still uses all of them.
- **Effect.** 75.0 % → **80.0 %** on dev-20 (query 90 → 100 %, action 60 → 60 %; no context overflow;
  `results/e3_medguardian_dev20_summary.json`). The 4 remaining dev failures (potassium and HbA1c tasks)
  are answer-format and order-decision differences whose expected form is not stated in the task text;
  tuning stopped here to avoid fitting the grader. E3 was then run once on the test split.

### 2026-11-10 — Gap 2 LLM baseline answered [] for every record
- **Observed.** With a "JSON only, answer [] if none" prompt, Qwen2.5-7B returned `[]` for all 30 test
  records in both LLM arms (F1 0.0; `results/e4_g2_test_v1_summary.json`) — a degenerate baseline.
- **Change.** Step-by-step prompt (procedure times → last/first dose gaps in hours → compare with
  interval → labs) with a `FINAL:` answer line, written and checked on the 10 dev records only, then frozen.
- **Effect.** Dev F1 0.0 → 0.44 for both LLM arms (LLM-only P 0.50 / R 0.40; LLM+RAG P 0.31 / R 0.80,
  false-alert rate 0.80; `results/e4_g2_dev_summary.json`). Test re-run once with the frozen prompt.
