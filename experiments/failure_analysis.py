"""Auto-bucket failed tasks from a run's trajectories into failure categories (handout 4.3).

    python experiments/failure_analysis.py e3_medguardian_test50 [e2_basepaper_test50 ...]

Categories (first match wins, in this order):
  context_overflow       conversation outgrew the context window
  round_limit            8 rounds used without FINISH
  invalid_action         response was not GET / POST / FINISH
  fhir_query_error       a GET failed (malformed search) and the task was not recovered
  unnecessary_post       a query task emitted a POST
  order_blocked          the verifier blocked the order (MedGuardian only)
  order_payload_or_value an action task emitted an order but was graded wrong
  no_order_or_wrong_value an action task emitted no order, or answered the value part wrongly
  wrong_answer           a query task finished with a wrong value
Writes results/failures_<run>.csv and results/failures_<run>_summary.json.
"""
import collections
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["context_overflow", "round_limit", "invalid_action", "fhir_query_error", "unnecessary_post", "order_blocked",
         "order_payload_or_value", "no_order_or_wrong_value", "wrong_answer"]


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
    if end["kind"] == "query":
        return "unnecessary_post" if end.get("n_post", 0) else "wrong_answer"
    if end.get("n_post_blocked", 0):
        return "order_blocked"
    return "order_payload_or_value" if end.get("n_post", 0) else "no_order_or_wrong_value"


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
