"""FHIR R4 tools. Same calls work on the MedAgentBench server (:8080) and our mock (:8081).

MedAgentBench search conventions: Observation?patient=<MRN>&code=<CODE>; Patient id == MRN.
"""
import os
import time
from datetime import date
from typing import Any, Dict, List, Optional

import requests

from ..state import PatientContext, ToolCallRecord

DEFAULT_BASE = os.environ.get("MOCK_FHIR_BASE", "http://localhost:8081/fhir/")
NOW = date(2023, 11, 13)  # MedAgentBench task "now" (2023-11-13T10:15:00+00:00)


def fhir_get(base: str, resource: str, params: Dict[str, Any], timeout: float = 15) -> Dict[str, Any]:
    url = base.rstrip("/") + "/" + resource
    params = dict(params)
    params["_format"] = "json"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _entries(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]


def age_on(birth_date: str, today: date = NOW) -> int:
    y, m, d = (int(p) for p in birth_date[:10].split("-"))
    return today.year - y - ((today.month, today.day) < (m, d))


def _obs_value(obs: Dict[str, Any]) -> Optional[float]:
    vq = obs.get("valueQuantity") or {}
    return vq.get("value")


def query_patient(ctx: PatientContext, base: str = DEFAULT_BASE, **search: Any) -> ToolCallRecord:
    """Search Patient by given/family/birthdate/identifier. Fills ctx when exactly one match."""
    t0 = time.time()
    try:
        patients = _entries(fhir_get(base, "Patient", {k: v for k, v in search.items() if v}))
        out = [{"id": p.get("id"),
                "name": " ".join((p.get("name") or [{}])[0].get("given", []) + [(p.get("name") or [{}])[0].get("family", "")]).strip(),
                "birthDate": p.get("birthDate"), "gender": p.get("gender")} for p in patients]
        if len(out) == 1:
            p = out[0]
            ctx.patient_id, ctx.name, ctx.birth_date, ctx.sex = p["id"], p["name"], p["birthDate"], p["gender"]
            if p["birthDate"]:
                ctx.age_years = age_on(p["birthDate"])
            out[0]["age_years"] = ctx.age_years
        rec = ToolCallRecord(tool_name="query_patient", arguments=search, output=out)
    except requests.RequestException as e:
        rec = ToolCallRecord(tool_name="query_patient", arguments=search, success=False, error=str(e))
    rec.latency_ms = (time.time() - t0) * 1000
    return ctx.record(rec)


def query_observation(ctx: PatientContext, code: str, patient_id: str = None,
                      base: str = DEFAULT_BASE) -> ToolCallRecord:
    """Latest Observation for a code (e.g. CREAT, WEIGHT, MG, K). Updates ctx labs."""
    pid = patient_id or ctx.patient_id
    args = {"patient": pid, "code": code}
    t0 = time.time()
    try:
        obs = _entries(fhir_get(base, "Observation", args))
        obs.sort(key=lambda o: o.get("effectiveDateTime", ""), reverse=True)
        out = [{"value": _obs_value(o), "unit": (o.get("valueQuantity") or {}).get("unit"),
                "effectiveDateTime": o.get("effectiveDateTime")} for o in obs]
        if out and out[0]["value"] is not None:
            latest = out[0]
            ctx.labs[code.upper()] = latest
            if code.upper() == "CREAT":
                ctx.serum_creatinine_mg_dl = float(latest["value"])
            elif code.upper() == "WEIGHT":
                ctx.weight_kg = float(latest["value"])
        rec = ToolCallRecord(tool_name="query_observation", arguments=args,
                             output={"count": len(out), "latest": out[0] if out else None})
    except requests.RequestException as e:
        rec = ToolCallRecord(tool_name="query_observation", arguments=args, success=False, error=str(e))
    rec.latency_ms = (time.time() - t0) * 1000
    return ctx.record(rec)
