"""E4-G2: process guardian vs non-timeline LLM baselines on MedGuardian-Timeline.

    python experiments/run_e4_g2.py --split test            # all three arms
    python experiments/run_e4_g2.py --split dev --arms llm_only --limit 2

Arms (each is a MedGuardianConfig switch combination, so this is also the Gap 2 ablation):
  llm_only       use_timeline_verifier=False, use_guideline_rag=False  -- Qwen reads the raw record
  llm_rag        use_timeline_verifier=False, use_guideline_rag=True   -- + retrieved guideline text
  guardian       use_timeline_verifier=True,  use_guideline_rag=True   -- MedGuardian (deterministic)

Metrics (micro over (case, type, item)): precision, recall, F1; false-alert rate (clean cases with
any flag); severity-weighted miss rate (weights high 3 / medium 2 / low 1).
Writes results/e4_g2_<split>_<arm>.csv and results/e4_g2_<split>_summary.json.
"""
import argparse
import csv
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.medguardian.guardian.process_guardian import audit  # noqa: E402
from src.medguardian.guardian.severity import WEIGHTS  # noqa: E402
from src.medguardian.llm import ChatModel  # noqa: E402
from src.medguardian.tools.guideline_rag import default_index  # noqa: E402
from src.medguardian.tracing import Tracer  # noqa: E402

TYPES = ("hold_too_short", "restart_too_early", "preop_lab_out_of_range", "missing_required_lab")
ARMS = ("llm_only", "llm_rag", "guardian")
MODEL = "medguardian-qwen"

PROMPT = """You are auditing the perioperative anticoagulation management of one patient.
Below is the patient's record as FHIR resources (JSON). Find every process discrepancy of these types:
- hold_too_short: the last dose of an anticoagulant/antiplatelet before the procedure was given too close to the procedure start
- restart_too_early: a direct oral anticoagulant was restarted too soon after the procedure ended
- preop_lab_out_of_range: a required pre-operative lab (INR or PLT) is out of the acceptable range
- missing_required_lab: a required pre-operative lab (INR or PLT) is missing or too old
{guidelines}
Record:
{record}

Answer with ONLY a JSON list, one object per discrepancy, e.g. [{{"type": "hold_too_short", "item": "apixaban"}}].
"item" is the drug name for interval problems and the lab code (INR or PLT) for lab problems. Answer [] if there are none."""


def compact_record(bundle):
    keep = []
    for e in bundle["entry"]:
        r = {k: v for k, v in e["resource"].items() if k not in ("subject", "identifier", "name")}
        keep.append(r)
    return json.dumps(keep, separators=(",", ":"))


def guideline_text(bundle):
    ix = default_index()
    drugs = sorted({(e["resource"].get("medicationCodeableConcept") or {}).get("text", "").lower()
                    for e in bundle["entry"] if e["resource"]["resourceType"] == "MedicationAdministration"})
    chunks = []
    for d in drugs:
        chunks += ix.search("%s stop before procedure hold restart after procedure" % d, k=3)
    chunks += ix.search("INR platelet count before surgery", k=2)
    seen, lines = set(), []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            lines.append("- " + c["text"])
    return "\nGuideline excerpts:\n" + "\n".join(lines) + "\n"


def parse_flags(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return [], False
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return [], False
    out = []
    for x in items if isinstance(items, list) else []:
        if isinstance(x, dict) and x.get("type") in TYPES:
            out.append({"type": x["type"], "item": str(x.get("item", "")).strip().lower()})
    return out, True


def key(flag):
    return flag["type"], flag["item"].lower()


def score(cases, preds):
    tp = fp = fn = 0
    false_alert = clean = 0
    miss_w = gold_w = 0
    for c in cases:
        g = {key(x): x for x in c["gold"]}
        p = {key(x) for x in preds[c["id"]]}
        tp += len(set(g) & p)
        fp += len(p - set(g))
        fn += len(set(g) - p)
        gold_w += sum(WEIGHTS[x["severity"]] for x in g.values())
        miss_w += sum(WEIGHTS[g[k]["severity"]] for k in set(g) - p)
        if not g:
            clean += 1
            false_alert += bool(p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n_cases": len(cases), "tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "false_alert_rate": round(false_alert / clean, 4) if clean else None,
            "severity_weighted_miss_rate": round(miss_w / gold_w, 4) if gold_w else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(os.path.join(ROOT, "data", "gap2", args.split + ".jsonl"))]
    cases = cases[:args.limit] if args.limit else cases
    llm = ChatModel(MODEL, temperature=args.temperature, seed=args.seed, max_tokens=512)
    tag = "" if args.temperature == 0 else "_t%g_s%d" % (args.temperature, args.seed)
    summary_path = os.path.join(ROOT, "results", "e4_g2_%s%s_summary.json" % (args.split, tag))
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    for arm in args.arms:
        preds, rows = {}, []
        tracer = Tracer(os.path.join(ROOT, "logs", "e4_g2_%s_%s%s.jsonl" % (args.split, arm, tag)), run_id="e4_g2_" + arm,
                        seed=args.seed)
        t_arm = time.time()
        for c in cases:
            t0 = time.time()
            if arm == "guardian":
                flags, parsed, raw, toks = audit(c["bundle"]), True, None, (0, 0)
            else:
                prompt = PROMPT.format(record=compact_record(c["bundle"]),
                                       guidelines=guideline_text(c["bundle"]) if arm == "llm_rag" else "")
                res = llm.chat([{"role": "user", "content": prompt}])
                flags, parsed = parse_flags(res.content)
                raw, toks = res.content, (res.tokens_in, res.tokens_out)
            preds[c["id"]] = flags
            g, p = {key(x) for x in c["gold"]}, {key(x) for x in flags}
            rows.append({"case_id": c["id"], "drug": c["drug"], "bleeding_risk": c["bleeding_risk"],
                         "gold": json.dumps(sorted(g)), "pred": json.dumps(sorted(p)), "tp": len(g & p),
                         "fp": len(p - g), "fn": len(g - p), "parsed": int(parsed), "tokens_in": toks[0],
                         "tokens_out": toks[1], "seconds": round(time.time() - t0, 2)})
            tracer.log("case", c["id"], arm=arm, gold=sorted(g), pred=flags, raw_response=raw)
            print("%s %s gold=%s pred=%s" % (arm, c["id"], sorted(g), sorted(p)), flush=True)
        tracer.close()
        with open(os.path.join(ROOT, "results", "e4_g2_%s_%s%s.csv" % (args.split, arm, tag)), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary[arm] = dict(score(cases, preds), seconds=round(time.time() - t_arm, 1),
                            parse_failures=sum(1 - r["parsed"] for r in rows))
        print(arm, summary[arm], flush=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
