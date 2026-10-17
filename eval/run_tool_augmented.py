"""Tool-augmented MedGuardian agent (LLM + FHIR + dosing tools) on the seed cases.

  python eval/run_tool_augmented.py [--cases SC01,SC02] [--sleep 6]
Writes eval/tool_augmented_results.md and outputs/tool_augmented_runs.jsonl.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from demo import request_text  # noqa: E402
from eval.mock_fhir_server import serve  # noqa: E402
from medguardian.llm_client import make_client  # noqa: E402
from medguardian.orchestrator import Orchestrator  # noqa: E402
from medguardian.state import to_dict  # noqa: E402


def grade(ctx, exp):
    r = ctx.recommendations[-1] if ctx.recommendations else None
    if r is None:
        return False, "no recommendation"
    ok = abs(r.crcl_ml_min - exp["crcl_ml_min"]) < 0.05 and r.adjusted_dose_mg == exp["adjusted_dose_mg"] \
        and r.contraindicated == exp["contraindicated"]
    return ok, "CrCl %s, %s mg, contra=%s" % (r.crcl_ml_min, r.adjusted_dose_mg, r.contraindicated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--sleep", type=float, default=6.0, help="pause between cases (free-tier rate limits)")
    a = ap.parse_args()
    cases = json.load(open(os.path.join(ROOT, "eval", "data", "seed_cases.json")))
    if a.cases:
        cases = [c for c in cases if c["case_id"] in a.cases.split(",")]
    serve(8081, background=True)
    llm = make_client()
    orch = Orchestrator(llm, "http://127.0.0.1:8081/fhir/")
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    log = open(os.path.join(ROOT, "outputs", "tool_augmented_runs.jsonl"), "w")
    rows, passed = [], 0
    for i, c in enumerate(cases):
        if i:
            time.sleep(a.sleep)
        try:
            res = orch.run(request_text(c))
            ctx = res["context"]
            ok, got = grade(ctx, c["expected"])
            chain = " → ".join(t.tool_name + ("" if t.success else "(err)") for t in ctx.tool_calls)
            status, lat, u = res["status"], res["latency_s"], res["usage"]
            log.write(json.dumps({"case": c["case_id"], "status": status, "pass": ok, "usage": u,
                                  "latency_s": lat, "trace": res["trace"], "final_answer": ctx.final_answer,
                                  "context": to_dict(ctx)}, default=str) + "\n")
        except Exception as e:  # record the failure (e.g. rate limit) instead of aborting the run
            ok, got, chain, status, lat, u = False, "error: %s" % str(e)[:80], "-", "error", "-", {}
        passed += ok
        rows.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c["case_id"], c["drug"], "PASS" if ok else "FAIL", got, c["expected"]["adjusted_dose_mg"],
            chain, lat, "%s/%s (%s calls)" % (u.get("prompt_tokens", "-"), u.get("completion_tokens", "-"), u.get("llm_calls", "-"))))
        print(rows[-1], flush=True)
    n = len(cases)
    md = ["# Tool-augmented MedGuardian agent: seed-case results", "",
          "- Model: `%s` via `%s` (OpenAI-compatible API), temperature 0" % (llm.model, llm.base_url),
          "- Date: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
          "- FHIR: mock server (`eval/mock_fhir_server.py`), synthetic patients only",
          "- Command: `python eval/run_tool_augmented.py`",
          "- **Result: %d/%d cases correct (%.0f%%)** (CrCl within 0.05 mL/min and exact adjusted dose / contraindication)" % (passed, n, 100.0 * passed / max(n, 1)),
          "", "| Case | Drug | Result | Agent output | Expected mg | Tool-call trace | Latency (s) | Tokens in/out |",
          "|---|---|---|---|---|---|---|---|"] + rows + [
          "", "Full traces (every ToolCallRecord and the final PatientContext) are in `outputs/tool_augmented_runs.jsonl` (not committed; regenerate with the command above)."]
    open(os.path.join(ROOT, "eval", "tool_augmented_results.md"), "w").write("\n".join(md) + "\n")
    print("\n%d/%d passed" % (passed, n))


if __name__ == "__main__":
    main()
