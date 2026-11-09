"""MedAgentBench runner: the upstream GET/POST/FINISH protocol against the live Docker FHIR server.

    python -m src.medguardian.runner --config experiments/configs/e2_basepaper.yaml --split dev --limit 5

`basepaper` mode (E0, E2) reproduces upstream `MedAgentBench.start_sample` step for step:
the same prompt (parsed from the submodule), the same action parsing, the same messages
injected back, the same 8-round cap and the same statuses, graded by upstream `refsol.py`.
It replaces the AgentBench controller/worker (Python 3.9 pin, a known upstream bug, and no
way to switch components for ablations) with an in-process loop; that is plan decision D1.

Outputs (named after the run):
  results/<run>.csv            one row per task (status, correct, rounds, tokens, latency)
  results/<run>_summary.json   overall / query / action success rate and status counts
  logs/<run>.jsonl             every step: raw response, parsed action, FHIR result, timing
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import mab
from .llm import ChatModel, ContextLimitError, LLMError
from .llm.pool import CHARS_PER_TOKEN_CEILING
from .tracing import Tracer, setup_logging

ROOT = mab.ROOT
SPLITS_DIR = os.path.join(ROOT, "data", "splits")
RESULTS_DIR = os.path.join(ROOT, "results")
LOGS_DIR = os.path.join(ROOT, "logs")
FHIR_CACHE_DIR = os.path.join(ROOT, ".cache", "fhir")

# Upstream AgentBench SampleStatus values.
COMPLETED = "completed"
CONTEXT_LIMIT = "agent context limit"
INVALID_ACTION = "agent invalid action"
LIMIT_REACHED = "task limit reached"
TASK_ERROR = "task error"

GET_NUDGE = ("Here is the response from the GET request:\n{data}. Please call FINISH if you have got "
             "answers for all the questions and finished all the requested tasks")
POST_OK = ("POST request accepted and executed successfully. Please call FINISH if you have got "
           "answers for all the questions and finished all the requested tasks")
POST_BAD = "Invalid POST request"

CSV_FIELDS = ("task_id", "category", "kind", "status", "correct", "result", "rounds", "n_get",
              "n_post", "tokens_in", "tokens_out", "latency_ms", "llm_cached_calls", "grader_error")


# -- protocol --------------------------------------------------------------------------------
def clean_response(content: str) -> str:
    """Upstream normalisation (strips Gemini-style code fences)."""
    return content.strip().replace("```tool_code", "").replace("```", "").strip()


def parse_action(content: str) -> Tuple[str, Any]:
    """Classify a response exactly as upstream does.

    Returns ("GET", url_with_format), ("POST", payload or None if not JSON),
    ("FINISH", result_string) or ("INVALID", None).
    """
    r = clean_response(content)
    if r.startswith("GET"):
        return "GET", r[3:].strip() + "&_format=json"
    if r.startswith("POST"):
        try:
            return "POST", json.loads("\n".join(r.split("\n")[1:]))
        except Exception:  # noqa: BLE001 - upstream catches everything here
            return "POST", None
    if r.startswith("FINISH("):
        return "FINISH", r[len("FINISH("):-1]
    return "INVALID", None


class FhirGetter:
    """Upstream `send_get_request`, memoised by URL (the FHIR server is read-only and deterministic)."""

    def __init__(self, cache_dir: Optional[str] = FHIR_CACHE_DIR):
        self.cache_dir = cache_dir
        self._send = mab.upstream_utils().send_get_request

    def get(self, url: str) -> Tuple[Dict[str, Any], bool]:
        path = None
        if self.cache_dir:
            key = hashlib.sha256(url.encode("utf-8")).hexdigest()
            path = os.path.join(self.cache_dir, key[:2], key + ".json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh), True
        res = self._send(url)
        if path and "data" in res:  # never cache transport errors
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False)
            os.replace(path + ".tmp", path)
        return res, False


def run_task(case: Dict[str, Any], funcs: List[Dict[str, Any]], llm: ChatModel, fhir: FhirGetter,
             cfg: Dict[str, Any], tracer: Tracer) -> Dict[str, Any]:
    """One task under the upstream base-paper protocol. Returns the CSV row."""
    tid = case["id"]
    fhir_base = cfg["fhir_base"]
    history: List[Tuple[str, str]] = [("user", mab.build_prompt(case, funcs, fhir_base))]
    status, result = LIMIT_REACHED, None
    tok_in = tok_out = n_get = n_post = cached_calls = 0
    latency = 0.0
    prefix = 0
    rounds = 0
    t_task = time.time()
    tracer.log("task_start", tid, category=mab.category(tid), kind=mab.task_kind(tid))
    try:
        for rnd in range(cfg["max_rounds"]):
            rounds = rnd + 1
            messages = [{"role": "assistant" if r == "agent" else "user", "content": c} for r, c in history]
            try:
                res = llm.chat(messages, prefix_tokens=prefix)
            except ContextLimitError as e:
                status = CONTEXT_LIMIT
                tracer.step(tid, round=rounds, success=False, result=str(e))
                break
            tok_in += res.tokens_in
            tok_out += res.tokens_out
            latency += res.latency_ms
            cached_calls += int(res.cached)
            history.append(("agent", res.content))
            kind, arg = parse_action(res.content)
            step = dict(round=rounds, raw_response=res.content, parsed_action=kind, latency_ms=res.latency_ms,
                        tokens_in=res.tokens_in, tokens_out=res.tokens_out, cached=res.cached)
            if kind == "GET":
                n_get += 1
                got, hit = fhir.get(arg)
                if "data" in got:
                    reply = GET_NUDGE.format(data=got["data"])
                    tracer.step(tid, tool="GET", args=arg, success=True,
                                result={"fhir_cached": hit, "chars": len(str(got["data"]))}, **step)
                else:
                    reply = "Error in sending the GET request: %s" % got["error"]
                    tracer.step(tid, tool="GET", args=arg, success=False, result=got["error"], **step)
            elif kind == "POST":
                n_post += 1
                reply = POST_OK if arg is not None else POST_BAD
                tracer.step(tid, tool="POST", args=arg, success=arg is not None, result=reply, **step)
            elif kind == "FINISH":
                status, result = COMPLETED, arg
                tracer.step(tid, tool="FINISH", args=arg, success=True, **step)
                break
            else:
                status = INVALID_ACTION
                tracer.step(tid, success=False, **step)
                break
            history.append(("user", reply))
            prefix = res.tokens_in + res.tokens_out + len(reply) // CHARS_PER_TOKEN_CEILING
    except LLMError as e:
        status, result = TASK_ERROR, None
        tracer.log("task_error", tid, error=str(e))

    correct, grader_out, grader_err = mab.grade(case, result, history, fhir_base)
    row = {"task_id": tid, "category": mab.category(tid), "kind": mab.task_kind(tid), "status": status,
           "correct": int(correct), "result": result, "rounds": rounds, "n_get": n_get, "n_post": n_post,
           "tokens_in": tok_in, "tokens_out": tok_out, "latency_ms": round(latency, 1),
           "llm_cached_calls": cached_calls, "grader_error": grader_err or ""}
    tracer.log("task_end", tid, wall_s=round(time.time() - t_task, 1), grader_stdout=grader_out.strip(),
               history=[{"role": r, "content": c} for r, c in history],
               **{k: v for k, v in row.items() if k != "task_id"})
    return row


# -- run orchestration -----------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("mode", "basepaper")
    cfg.setdefault("base_url", "http://localhost:11434/v1")
    cfg.setdefault("fhir_base", mab.DEFAULT_FHIR_BASE)
    cfg.setdefault("max_rounds", 8)
    cfg.setdefault("temperature", 0.0)
    cfg.setdefault("max_tokens", 2048)
    cfg.setdefault("seed", 0)
    cfg.setdefault("num_ctx", 16384)
    if cfg["mode"] != "basepaper":
        raise NotImplementedError("mode %r is not implemented yet" % cfg["mode"])
    return cfg


def select_tasks(cases: List[Dict[str, Any]], split: str, limit: Optional[int] = None,
                 per_category: Optional[int] = None, task_ids: Optional[List[str]] = None):
    by_id = {c["id"]: c for c in cases}
    if task_ids:
        ids = list(task_ids)
    elif split == "all":
        ids = [c["id"] for c in cases]
    else:
        with open(os.path.join(SPLITS_DIR, split + ".json"), encoding="utf-8") as fh:
            ids = json.load(fh)["ids"]
    if per_category:
        seen: Counter = Counter()
        kept = []
        for tid in ids:
            if seen[mab.category(tid)] < per_category:
                kept.append(tid)
                seen[mab.category(tid)] += 1
        ids = kept
    if limit:
        ids = ids[:limit]
    missing = [t for t in ids if t not in by_id]
    if missing:
        raise KeyError("unknown task ids: %s" % missing)
    return [by_id[t] for t in ids]


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def sr(sub):
        return round(100.0 * sum(r["correct"] for r in sub) / len(sub), 2) if sub else None

    query = [r for r in rows if r["kind"] == "query"]
    action = [r for r in rows if r["kind"] == "action"]
    lat = [r["latency_ms"] / 1000 for r in rows]
    return {"n": len(rows), "n_query": len(query), "n_action": len(action),
            "overall_sr": sr(rows), "query_sr": sr(query), "action_sr": sr(action),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "by_category": {c: sr([r for r in rows if r["category"] == c])
                            for c in mab.CATEGORIES if any(r["category"] == c for r in rows)},
            "tokens_in": sum(r["tokens_in"] for r in rows), "tokens_out": sum(r["tokens_out"] for r in rows),
            "llm_seconds_per_task_median": round(statistics.median(lat), 1) if lat else None}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="Run MedAgentBench tasks against the local FHIR server.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="dev", help="dev | test | all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-category", type=int, help="keep at most N tasks per category")
    ap.add_argument("--task-ids", nargs="*")
    ap.add_argument("--run-name", help="default: <config name>_<split>")
    ap.add_argument("--no-cache", action="store_true", help="bypass the LLM response cache")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    log = setup_logging(args.log_level)
    cfg = load_config(args.config)
    run = args.run_name or "%s_%s" % (cfg["name"], args.split)
    cases = select_tasks(mab.load_cases(), args.split, args.limit, args.per_category, args.task_ids)
    funcs = mab.load_funcs()
    mab.refsol()  # fail fast if the grader is missing
    llm = ChatModel(model=cfg["model"], base_url=cfg["base_url"], num_ctx=cfg["num_ctx"],
                    temperature=cfg["temperature"], max_tokens=cfg["max_tokens"], seed=cfg["seed"],
                    cache_dir=None if args.no_cache else os.path.join(ROOT, ".cache", "llm"))
    fhir = FhirGetter()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, run + ".csv")
    log_path = os.path.join(LOGS_DIR, run + ".jsonl")
    if os.path.exists(log_path):  # a re-run replays from cache; start a fresh trajectory file
        os.replace(log_path, log_path + ".prev")
    log.info("run %s: %d tasks, model=%s, temp=%s, seed=%s", run, len(cases), cfg["model"],
             cfg["temperature"], cfg["seed"])

    rows = []
    with Tracer(log_path, run_id=run, seed=cfg["seed"]) as tracer, \
            open(csv_path, "w", newline="", encoding="utf-8") as fh:
        tracer.log("run_start", config=cfg, split=args.split, n_tasks=len(cases),
                   started=dt.datetime.now().isoformat(timespec="seconds"))
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i, case in enumerate(cases, 1):
            t0 = time.time()
            row = run_task(case, funcs, llm, fhir, cfg, tracer)
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            log.info("[%d/%d] %s %s correct=%d rounds=%d tokens_in=%d %.0fs%s", i, len(cases), row["task_id"],
                     row["status"], row["correct"], row["rounds"], row["tokens_in"], time.time() - t0,
                     " (cached)" if row["llm_cached_calls"] == row["rounds"] and row["rounds"] else "")
        summary = summarise(rows)
        summary.update({"run": run, "config": cfg, "split": args.split,
                        "finished": dt.datetime.now().isoformat(timespec="seconds")})
        tracer.log("run_end", summary=summary)
    with open(os.path.join(RESULTS_DIR, run + "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    log.info("SR overall=%s query=%s action=%s | statuses %s", summary["overall_sr"], summary["query_sr"],
             summary["action_sr"], summary["status_counts"])
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
