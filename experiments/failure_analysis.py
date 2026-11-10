"""Auto-bucket failed tasks from a run's trajectories into failure categories (handout 4.3).

    python experiments/failure_analysis.py e3_medguardian_test50 [e2_basepaper_test50 ...]

Categories (first match wins, in this order):
  context_overflow       conversation outgrew the context window
  round_limit            8 rounds used without FINISH
  invalid_action         response was not GET / POST / FINISH
  fhir_query_error       a GET failed (malformed search) and the task was not recovered
  unnecessary_post       a query task emitted a POST
  order_blocked          the verifier blocked the order (MedGuardian only)
  answer_format          the right value was given, but not in the expected [value, ...] form
  time_window_error      a value outside the requested window was reported (or -1 although one existed)
  order_payload_or_value an action task emitted an order but was graded wrong
  order_decision         an action task emitted no order and its value part was not the issue
  wrong_value            any other wrong final value
The last five use the grader's printed reference value, for post-hoc analysis only.
Writes results/failures_<run>.csv and results/failures_<run>_summary.json.
"""
import collections
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["context_overflow", "round_limit", "invalid_action", "fhir_query_error", "unnecessary_post", "order_blocked",
         "answer_format", "time_window_error", "order_payload_or_value", "order_decision", "wrong_value"]


def _numbers(x):
    """All numbers in a value (list / dict / string)."""
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(x) if not isinstance(x, str) else x)]


def _reference(grader_stdout):
    """The grader's printed reference list, e.g. 'task4_1 [-1] [2.3]' -> [-1] (post-hoc analysis only)."""
    m = re.match(r"\S+\s+(\[.*?\])\s", grader_stdout or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1).replace("'", '"'))
    except ValueError:
        return None


def refine(end):
    """Split a completed-but-wrong task using the grader's printed reference value."""
    ref, result = _reference(end.get("grader_stdout")), end.get("result") or ""
    try:
        ans = json.loads(result)
    except ValueError:
        ans = result
    if ref and isinstance(ref[0], (int, float)):
        r0, got = float(ref[0]), _numbers(ans)
        if any(abs(g - r0) <= 0.05 for g in got) and ans != ref:
            return "answer_format"          # right value, not in the expected [value, ...] form
        if r0 == -1 and got and -1 not in got:
            return "time_window_error"      # reported a value although none was in the window
        if got == [-1.0] and r0 != -1:
            return "time_window_error"      # reported 'none in window' although one existed
        return "wrong_value"
    return None


def bucket(end, steps):
    st = end["status"]
    if st == "agent context limit":
        return "context_overflow"
    if st == "task limit reached":
        return "round_limit"
    if st in ("agent invalid action", "task error"):
        return "invalid_action"
    get_errors = [s for s in steps if s.get("tool") == "GET" and not s.get("success")]
    if get_errors and not [s for s in steps if s.get("tool") == "GET" and s.get("success")]:
        return "fhir_query_error"
    if end["kind"] == "query" and end.get("n_post", 0):
        return "unnecessary_post"
    if end.get("n_post_blocked", 0):
        return "order_blocked"
    refined = refine(end)
    if refined:
        return refined
    if end["kind"] == "action":
        return "order_payload_or_value" if end.get("n_post", 0) else "order_decision"
    return "wrong_value"


def analyse(run):
    path = os.path.join(ROOT, "logs", run + ".jsonl")
    steps, rows = collections.defaultdict(list), []
    for line in open(path):
        e = json.loads(line)
        if e["event"] == "step":
            steps[e["task_id"]].append(e)
        elif e["event"] == "task_end" and not e["correct"]:
            rows.append({"task_id": e["task_id"], "category": e["category"], "kind": e["kind"], "status": e["status"],
                         "failure": bucket(e, steps[e["task_id"]]), "rounds": e["rounds"]})
    counts = collections.Counter(r["failure"] for r in rows)
    summary = {"run": run, "n_failed": len(rows), "counts": {k: counts.get(k, 0) for k in ORDER if counts.get(k)}}
    with open(os.path.join(ROOT, "results", "failures_%s.csv" % run), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task_id", "category", "kind", "status", "failure", "rounds"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(ROOT, "results", "failures_%s_summary.json" % run), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    return summary


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(analyse(r)))
