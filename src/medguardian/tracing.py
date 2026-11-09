"""Structured JSONL trajectory logging.

One JSON object per line. Every record carries run_id, task_id and an event type; step records
add round, raw_response, parsed_action, args, result, success, latency_ms and token counts.
The format is append-only so an interrupted run keeps every completed step.
"""
import json
import logging
import os
import time
from typing import Any, Dict, Optional

STEP_FIELDS = ("round", "raw_response", "parsed_action", "tool", "args", "result", "success",
               "latency_ms", "tokens_in", "tokens_out", "cached")


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("medguardian")


class Tracer:
    """Append-only JSONL writer for one run."""

    def __init__(self, path: str, run_id: str, seed: Optional[int] = None):
        self.path = path
        self.run_id = run_id
        self.seed = seed
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, event: str, task_id: Optional[str] = None, **fields: Any) -> Dict[str, Any]:
        rec = {"ts": round(time.time(), 3), "run_id": self.run_id, "seed": self.seed,
               "task_id": task_id, "event": event}
        rec.update(fields)
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        return rec

    def step(self, task_id: str, **fields: Any) -> Dict[str, Any]:
        unknown = set(fields) - set(STEP_FIELDS)
        if unknown:
            raise ValueError("unknown step fields: %s" % sorted(unknown))
        return self.log("step", task_id, **fields)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_trace(path: str):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
