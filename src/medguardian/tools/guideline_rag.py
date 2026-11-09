"""BM25 guideline retrieval (plan D5). Every chunk carries a citation and machine-readable facts.

The process guardian never hard-codes a threshold: it retrieves the chunk for the drug / lab in
question and reads the numbers from that chunk, so every flag it raises can cite its source.
"""
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CORPUS = os.path.join(ROOT, "data", "corpora", "guidelines", "periop_anticoagulation.jsonl")


def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class GuidelineIndex:
    def __init__(self, path: str = CORPUS):
        with open(path, encoding="utf-8") as fh:
            self.chunks = [json.loads(l) for l in fh if l.strip()]
        self.bm25 = BM25Okapi([_tok(c["text"] + " " + c["id"].replace("-", " ")) for c in self.chunks])

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in order]

    def resolve_drug(self, text: str) -> Optional[str]:
        """Map free medication text ("ELIQUIS 5 mg tablet", "apixaban") to the generic name the
        corpus uses, via retrieval over the drug chunks (generic or brand name must appear)."""
        toks = set(_tok(text))
        for c in self.search(text + " stop before procedure", k=5):
            f = c["facts"]
            if f.get("kind") == "hold" and (f["drug"] in toks or toks & set(f.get("brand_names", []))):
                return f["drug"]
        return None

    def bleeding_risk(self, procedure: str) -> Optional[Dict[str, Any]]:
        """Classify a procedure's bleeding risk from the retrieved classification chunk."""
        toks = set(_tok(procedure))
        for c in self.search("bleeding risk classification " + procedure, k=len(self.chunks)):
            f = c["facts"]
            if f.get("kind") == "bleeding_risk":
                for level in ("high", "low"):
                    if toks & set(f["keywords"][level]):
                        return {"level": level, "chunk": c, "inferred": True}
                return {"level": f.get("default", "high"), "chunk": c, "inferred": True, "default": True}
        return None

    def hold_rule(self, drug: str, crcl: Optional[float] = None) -> Optional[Dict[str, Any]]:
        q = "%s stop before procedure hold interruption last dose hours" % drug
        if drug == "dabigatran" and crcl is not None:
            q += " creatinine clearance %s" % ("below 50 reduced renal" if crcl < 50 else "50 or more")
        for c in self.search(q, k=5):
            f = c["facts"]
            if f.get("kind") == "hold" and f.get("drug") == drug and \
                    (crcl is None or (crcl >= f.get("crcl_min", -1e9) and crcl < f.get("crcl_max", 1e9))):
                return c
        return None

    def resume_rule(self, drug: str) -> Optional[Dict[str, Any]]:
        for c in self.search("resume restart %s after procedure hours haemostasis" % drug, k=5):
            if c["facts"].get("kind") == "resume" and drug in c["facts"].get("drugs", []):
                return c
        return None

    def lab_rules(self, drugs: List[str]) -> List[Dict[str, Any]]:
        out = []
        for lab_query in ("INR before surgery warfarin", "platelet count before invasive procedure"):
            for c in self.search(lab_query, k=3):
                f = c["facts"]
                if f.get("kind") == "lab" and (f["applies_to"] == ["all"] or set(f["applies_to"]) & set(drugs)):
                    out.append(c)
                    break
        return out


@lru_cache(maxsize=1)
def default_index() -> GuidelineIndex:
    return GuidelineIndex()
