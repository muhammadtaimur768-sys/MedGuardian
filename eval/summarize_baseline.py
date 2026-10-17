"""Summarise the upstream MedAgentBench baseline run into eval/baseline_results.md.

Reads medagentbench/outputs/MedAgentBenchv1/<agent>/<task>/overall.json (written by the upstream
assigner; `raw_results[i].index` indexes the task's data file, `status` ends in Correct/Incorrect)
plus error.jsonl. If nothing has been run yet, writes a "pending" results file with the run steps.
"""
import datetime
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAB = os.path.join(ROOT, "medagentbench")
OUT_DIRS = [os.path.join(MAB, "outputs", "MedAgentBenchv1"), os.path.join(ROOT, "outputs", "MedAgentBenchv1")]
SUBSET_DATA = os.path.join(MAB, "data", "medagentbench", "medguardian_subset_30.json")
FULL_DATA = os.path.join(MAB, "data", "medagentbench", "test_data_v2.json")
RESULTS_MD = os.path.join(ROOT, "eval", "baseline_results.md")
QUERY = ["task1", "task2", "task4", "task6", "task7"]
ACTION = ["task3", "task5", "task8", "task9", "task10"]
RATE_LIMIT = re.compile(r"429|rate.?limit|RESOURCE_EXHAUSTED|Too Many Requests|quota", re.I)
PAPER = "Claude 3.5 Sonnet v2 (Jiang et al., NEJM AI 2025): query 85.33%, action 54.00%, overall 69.67% (300 tasks)"
RUN_STEPS = """```bash
python3 eval/run_baseline.py setup     # subset data + overlay configs + generated Gemini agent yaml
docker run -p 8080:8080 jyxsu6/medagentbench:latest
python3 eval/run_baseline.py check     # FHIR server + refsol.py present
python3 eval/run_baseline.py start     # controller + 1 worker (leave running)
cd medagentbench && python -m src.assigner --config configs/assignments/medguardian_gemini.yaml
python3 eval/summarize_baseline.py
```"""


def model_id():
    gen = os.path.join(MAB, "configs", "agents", "gemini-baseline.generated.yaml")
    if os.path.exists(gen):
        m = re.search(r'model:\s*"?([^"\n]+)"?', open(gen).read())
        if m:
            return m.group(1).strip()
    return os.environ.get("LLM_MODEL", "unknown")


def pct(c, n):
    return "%.2f%% (%d/%d)" % (100.0 * c / n, c, n) if n else "n/a"


def summarise(overall_path):
    task_dir = os.path.dirname(overall_path)
    agent, task = os.path.basename(os.path.dirname(task_dir)), os.path.basename(task_dir)
    data_file = SUBSET_DATA if "subset" in task and os.path.exists(SUBSET_DATA) else FULL_DATA
    ids = [r["id"] for r in json.load(open(data_file))]
    overall = json.load(open(overall_path))
    per_type, statuses, rate_limited = {}, {}, 0
    for r in overall.get("raw_results", []):
        ttype = ids[r["index"]].split("_")[0]
        status = r.get("status", "")
        correct = status.endswith("Correct") and not status.endswith("Incorrect")
        c, n = per_type.get(ttype, (0, 0))
        per_type[ttype] = (c + int(correct), n + 1)
        base = re.sub(r"(Correct|Incorrect)$", "", status)
        statuses[base] = statuses.get(base, 0) + 1
        if RATE_LIMIT.search(json.dumps(r.get("history") or [])):
            rate_limited += 1
    err_file = os.path.join(task_dir, "error.jsonl")
    errors = []
    if os.path.exists(err_file):
        errors = [line for line in open(err_file) if line.strip()]
    rate_limited += sum(1 for e in errors if RATE_LIMIT.search(e))

    def group(types):
        c = sum(per_type.get(t, (0, 0))[0] for t in types)
        n = sum(per_type.get(t, (0, 0))[1] for t in types)
        return pct(c, n)

    when = datetime.datetime.fromtimestamp(os.path.getmtime(overall_path)).strftime("%Y-%m-%d %H:%M")
    lines = [
        "## Agent `%s` on task `%s`" % (agent, task), "",
        "- Model ID: `%s` (Gemini OpenAI-compatible endpoint, temperature 0, reasoning_effort low)" % model_id(),
        "- Run date: %s" % when,
        "- Subset: %s (%d tasks)" % (os.path.basename(data_file), len(ids)),
        "- Upstream `success rate`: %.2f%%" % (100.0 * overall.get("success rate", 0)), "",
        "| Split | Success rate |", "|---|---|",
        "| **Overall** | %s |" % group(QUERY + ACTION),
        "| Query (task1,2,4,6,7) | %s |" % group(QUERY),
        "| Action (task3,5,8,9,10) | %s |" % group(ACTION),
    ]
    for t in sorted(per_type, key=lambda s: int(s[4:])):
        lines.append("| %s (%s) | %s |" % (t, "query" if t in QUERY else "action", pct(*per_type[t])))
    lines += ["", "Status counts (upstream SampleStatus): " + ", ".join("%s=%d" % kv for kv in sorted(statuses.items())),
              "Errors in error.jsonl: %d; samples with rate-limit (429/quota) signs: %d" % (len(errors), rate_limited), ""]
    return lines


def main():
    found = []
    for d in OUT_DIRS:
        found += sorted(glob.glob(os.path.join(d, "*", "*", "overall.json")))
    header = ["# Upstream MedAgentBench baseline results", "",
              "Unmodified MedAgentBench HTTPAgent (no MedGuardian tools) on a stratified subset: first 3 tasks of",
              "each of the 10 task types = 30 tasks (`eval/run_baseline.py setup`). Graded by upstream refsol.py.", "",
              "Paper reference: " + PAPER + ". Our numbers are on 30 tasks with a different model, so they are",
              "not directly comparable; they are an initial baseline only.", ""]
    if not found:
        body = ["## Status: pending: not yet run", "",
                "No `overall.json` found under `medagentbench/outputs/MedAgentBenchv1/`. Steps:", "", RUN_STEPS, ""]
    else:
        body = []
        for p in found:
            body += summarise(p)
        body += ["### How to reproduce", "", RUN_STEPS, ""]
    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(header + body))
    print("wrote %s (%s)" % (os.path.relpath(RESULTS_MD, ROOT), "%d run(s)" % len(found) if found else "pending"))


if __name__ == "__main__":
    sys.exit(main())
