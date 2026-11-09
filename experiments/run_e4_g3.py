"""E4-G3: sonography report fusion -- image-only vs dictation-only vs fused (H3: fused F1 > both).

    python experiments/run_e4_g3.py --split test

Per case: Apple Vision OCR on the screenshot, faster-whisper on the dictation audio, Qwen fills the
template from the transcript, then template-constrained fusion.
Metrics: field-level P/R/F1 (numeric match within 0.05, categorical exact), measurement
transcription error rate (wrong numeric values / numeric values output for gold fields), conflict
detection P/R (fused arm), BLEU-4 of the rendered report vs the reference report, ASR WER.
Writes results/e4_g3_<split>.csv, results/e4_g3_<split>_summary.json, logs/e4_g3_<split>.jsonl.
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import sacrebleu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.medguardian.llm import ChatModel  # noqa: E402
from src.medguardian.tracing import Tracer  # noqa: E402
from src.sonography import pipeline  # noqa: E402
from src.sonography.templates import NUMERIC_FIELDS, render_report  # noqa: E402

ARMS = ("image_only", "dictation_only", "fused")
CACHE = os.path.join(ROOT, ".cache", "gap3")


def field_match(field, pred, gold):
    if field in NUMERIC_FIELDS:
        return isinstance(pred, (int, float)) and abs(pred - gold) <= 0.05
    return pred == gold


def words(s):
    s = s.lower().replace("centimeters", "cm").replace("millimeters", "mm")
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", s)


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def perceive(case):
    """OCR + ASR, cached on disk (deterministic)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, case["id"] + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    image, lines = pipeline.ocr_measurements(os.path.join(ROOT, case["image"]))
    t1 = time.time()
    transcript = pipeline.transcribe(os.path.join(ROOT, case["audio"]))
    t2 = time.time()
    out = {"image": image, "ocr_lines": lines, "transcript": transcript,
           "ocr_seconds": round(t1 - t0, 2), "asr_seconds": round(t2 - t1, 2)}
    json.dump(out, open(path, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(os.path.join(ROOT, "data", "gap3", args.split + ".jsonl"))]
    cases = cases[:args.limit] if args.limit else cases
    llm = ChatModel("medguardian-qwen", temperature=0.0, seed=0, max_tokens=400)
    tracer = Tracer(os.path.join(ROOT, "logs", "e4_g3_%s.jsonl" % args.split), run_id="e4_g3", seed=0)
    agg = {a: {"tp": 0, "fp": 0, "fn": 0, "num_out": 0, "num_wrong": 0, "hyps": []} for a in ARMS}
    refs, rows = [], []
    conf_tp = conf_fp = conf_fn = 0
    wers = []
    for c in cases:
        p = perceive(c)
        dictation, raw = pipeline.extract_dictation(p["transcript"], llm)
        fused, conflicts = pipeline.fuse(p["image"], dictation)
        preds = {"image_only": p["image"], "dictation_only": dictation, "fused": fused}
        gold = c["gold"]
        refs.append(c["reference_report"])
        w = wer(c["script"], p["transcript"])
        wers.append(w)
        row = {"case_id": c["id"], "voice": c["voice"], "asr_wer": round(w, 4), "gold_conflicts": json.dumps(c["conflicts"]),
               "flagged_conflicts": json.dumps([x["field"] for x in conflicts])}
        for arm, pred in preds.items():
            a = agg[arm]
            ok = {f for f, v in pred.items() if f in gold and field_match(f, v, gold[f])}
            a["tp"] += len(ok)
            a["fp"] += len(pred) - len(ok)
            a["fn"] += len(gold) - len(ok)
            nums = [f for f in pred if f in NUMERIC_FIELDS and f in gold]
            a["num_out"] += len(nums)
            a["num_wrong"] += sum(1 for f in nums if f not in ok)
            a["hyps"].append(render_report(pred))
            row[arm + "_correct"] = len(ok)
            row[arm + "_predicted"] = len(pred)
        row["gold_fields"] = len(gold)
        flagged, gold_c = {x["field"] for x in conflicts}, set(c["conflicts"])
        conf_tp += len(flagged & gold_c)
        conf_fp += len(flagged - gold_c)
        conf_fn += len(gold_c - flagged)
        rows.append(row)
        tracer.log("case", c["id"], ocr_lines=p["ocr_lines"], transcript=p["transcript"], script=c["script"],
                   image=p["image"], dictation=dictation, llm_raw=raw, fused=fused, conflicts=conflicts, gold=gold)
        print(c["id"], {k: v for k, v in row.items() if k.endswith("_correct")}, "of", len(gold), "| wer %.3f" % w, flush=True)
    tracer.close()
    summary = {"n_cases": len(cases), "asr_wer_mean": round(sum(wers) / len(wers), 4)}
    for arm, a in agg.items():
        prec = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0.0
        rec = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0.0
        summary[arm] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                        "measurement_error_rate": round(a["num_wrong"] / a["num_out"], 4) if a["num_out"] else None,
                        "bleu4": round(sacrebleu.corpus_bleu(a["hyps"], [refs]).score, 2)}
    summary["fused"]["conflict_precision"] = round(conf_tp / (conf_tp + conf_fp), 4) if conf_tp + conf_fp else None
    summary["fused"]["conflict_recall"] = round(conf_tp / (conf_tp + conf_fn), 4) if conf_tp + conf_fn else None
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "e4_g3_%s.csv" % args.split), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(ROOT, "results", "e4_g3_%s_summary.json" % args.split), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
