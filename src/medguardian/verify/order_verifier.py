"""Gap 1 pre-order verifier: every POST is checked before it reaches the EHR.

Checks, in order:
  1. structure  -- endpoint is a writable resource, resourceType matches it, required fields of
                   the published function schema (funcs_v1.json) are present;
  2. identity   -- the order is for the patient named in the task;
  3. coding     -- codes stated in the task (NDC, LOINC, SNOMED, flowsheet ID) are used with the
                   right code system; status/intent values are the supported ones;
  4. timing     -- dates are ISO-8601; "next day at 8am" style requests resolve to that instant;
  5. indication -- the triggering lab is looked up deterministically on the FHIR server and the
                   order must be indicated by the rule written in the task (e.g. "if low");
  6. dose       -- the dose is recomputed by a calculator from the protocol written in the task
                   (tiered magnesium, per-0.1 mEq/L potassium) and must match, with units.

Every rule is derived from the task's own text, the published FHIR function schemas, or live
FHIR data -- never from the benchmark's grader. A draft that fails is NOT sent; the agent gets the
list of problems and one repair attempt. A draft for an order that is not indicated is blocked.
"""
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .. import mab
from ..tools.fhir_summary import _iso, latest_observation
from ..tools.task_context import TaskFacts, next_day_at

WRITABLE = ("Observation", "MedicationRequest", "ServiceRequest")
SYSTEMS = {"ndc": "http://hl7.org/fhir/sid/ndc", "loinc": "http://loinc.org", "snomed": "http://snomed.info/sct"}


@dataclass
class Verdict:
    ok: bool
    issues: List[str] = field(default_factory=list)
    not_indicated: bool = False      # the order should not be placed at all
    evidence: Dict[str, Any] = field(default_factory=dict)


def _required_fields() -> Dict[str, List[str]]:
    out = {}
    for f in mab.load_funcs():
        name = f["name"]
        if name.startswith("POST"):
            out[name.rsplit("/", 1)[-1]] = list(f.get("parameters", {}).get("required", []))
    return out


_REQUIRED: Optional[Dict[str, List[str]]] = None


def required_fields() -> Dict[str, List[str]]:
    global _REQUIRED
    if _REQUIRED is None:
        _REQUIRED = _required_fields()
    return _REQUIRED


def _codings(cc: Any) -> List[Dict[str, Any]]:
    return (cc or {}).get("coding") or [] if isinstance(cc, dict) else []


def _has_code(cc: Any, code: str, system: str) -> bool:
    return any(c.get("code") == code and c.get("system") == system for c in _codings(cc))


def _first(lst: Any) -> Dict[str, Any]:
    return lst[0] if isinstance(lst, list) and lst and isinstance(lst[0], dict) else {}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _mg_dose(tiers, value: float):
    for lo, hi, hi_inclusive, grams, hours in tiers:
        if value >= lo and (value <= hi if hi_inclusive else value < hi):
            return grams, hours
    return None


def verify(url: str, payload: Any, facts: TaskFacts, get: Callable, fhir_base: str) -> Verdict:
    v = Verdict(ok=True)
    issues = v.issues
    endpoint = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    if payload is None:
        issues.append("The payload is not valid JSON.")
        v.ok = False
        return v
    if not isinstance(payload, dict):
        issues.append("The payload must be a single JSON object.")
        v.ok = False
        return v
    if endpoint not in WRITABLE:
        issues.append("POST must go to one of %s, not '%s'." % (", ".join("{api_base}" + w for w in WRITABLE), endpoint))
    elif payload.get("resourceType") != endpoint:
        issues.append("resourceType is %r but the URL is %s." % (payload.get("resourceType"), endpoint))
    for f in required_fields().get(endpoint, []):
        if f not in payload:
            issues.append("Required field '%s' is missing." % f)

    ref = (payload.get("subject") or {}).get("reference") if isinstance(payload.get("subject"), dict) else None
    if facts.mrn and ref != "Patient/%s" % facts.mrn:
        issues.append("subject.reference must be 'Patient/%s' (got %r)." % (facts.mrn, ref))

    for k in ("effectiveDateTime", "authoredOn", "occurrenceDateTime"):
        if k in payload and not _iso(str(payload[k])):
            issues.append("%s must be an ISO-8601 date-time (got %r)." % (k, payload[k]))

    if endpoint == "Observation":
        if payload.get("status") != "final":
            issues.append("Observation.status must be 'final'.")
        if facts.flowsheet and (payload.get("code") or {}).get("text") != facts.flowsheet:
            issues.append("code.text must be the flowsheet ID '%s'." % facts.flowsheet)
        if not str(payload.get("valueString") or "").strip():
            issues.append("valueString must contain the measured value.")

    elif endpoint == "MedicationRequest":
        if facts.ndc and not _has_code(payload.get("medicationCodeableConcept"), facts.ndc, SYSTEMS["ndc"]):
            issues.append("medicationCodeableConcept.coding must include code '%s' with system '%s'."
                          % (facts.ndc, SYSTEMS["ndc"]))
        if payload.get("status") not in (None, "active"):
            issues.append("status should be 'active'.")
        if payload.get("intent") not in (None, "order"):
            issues.append("intent should be 'order'.")
        dr = _first(_first(payload.get("dosageInstruction")).get("doseAndRate"))
        dose = dr.get("doseQuantity") or {}
        if not isinstance(dose.get("value"), (int, float)) or not dose.get("unit"):
            issues.append("dosageInstruction[0].doseAndRate[0].doseQuantity needs a numeric value and a unit.")
        _check_indication_and_dose(v, endpoint, payload, facts, get, fhir_base, dose, dr)

    elif endpoint == "ServiceRequest":
        code = payload.get("code")
        if facts.snomed and not _has_code(code, facts.snomed, SYSTEMS["snomed"]):
            issues.append("code.coding must include SNOMED '%s' with system '%s'." % (facts.snomed, SYSTEMS["snomed"]))
        if facts.loinc and not facts.snomed and not _has_code(code, facts.loinc, SYSTEMS["loinc"]):
            issues.append("code.coding must include LOINC '%s' with system '%s'." % (facts.loinc, SYSTEMS["loinc"]))
        for k, want in (("status", "active"), ("intent", "order"), ("priority", "stat")):
            if payload.get(k) not in (None, want):
                issues.append("%s should be '%s'." % (k, want))
        if facts.free_text:
            note = payload.get("note")
            text = note.get("text") if isinstance(note, dict) else (_first(note).get("text") if isinstance(note, list) else None)
            if _norm(facts.free_text) not in _norm(text or ""):
                issues.append("note.text must contain the requested free text verbatim.")
        if facts.next_day_hour is not None and facts.now:
            want = next_day_at(facts.now, facts.next_day_hour)
            got = _iso(str(payload.get("occurrenceDateTime") or ""))
            if got != want:
                issues.append("occurrenceDateTime must be %s." % want.isoformat())
        _check_indication_and_dose(v, endpoint, payload, facts, get, fhir_base, None, None)

    v.ok = v.ok and not issues
    return v


def _check_indication_and_dose(v: Verdict, endpoint: str, payload: Dict[str, Any], facts: TaskFacts,
                               get: Callable, fhir_base: str, dose: Optional[Dict[str, Any]],
                               dr: Optional[Dict[str, Any]]) -> None:
    """Indication (from the lab) and dose (from the task's protocol)."""
    if not facts.mrn:
        return
    code_for = facts.lab_codes
    if facts.mg_tiers and "magnesium" in code_for:
        lab = latest_observation(get, fhir_base, facts.mrn, code_for["magnesium"])
        in_window = lab and facts.now and facts.window_hours and \
            0 <= (facts.now - lab["time"]).total_seconds() / 3600 <= facts.window_hours
        v.evidence["magnesium"] = lab and {"value": lab["value"], "time": lab["time"].isoformat()}
        if not in_window:
            _block(v, "no magnesium result in the last %g hours" % (facts.window_hours or 0))
            return
        tier = _mg_dose(facts.mg_tiers, lab["value"])
        if tier is None:
            _block(v, "latest magnesium %.2f mg/dL is not low" % lab["value"])
            return
        if endpoint == "MedicationRequest" and dose is not None:
            grams, hours = tier
            if dose.get("value") != grams or str(dose.get("unit", "")).lower() not in ("g", "gram", "grams"):
                v.issues.append("Dose must be %g g for magnesium %.2f mg/dL (task protocol)." % (grams, lab["value"]))
            rate = (dr or {}).get("rateQuantity") or {}
            if rate and (rate.get("value") != hours or str(rate.get("unit", "")).lower() not in ("h", "hr", "hour", "hours")):
                v.issues.append("Infuse over %g h (rateQuantity value %g, unit 'h')." % (hours, hours))
    elif facts.k_rule and "potassium" in code_for:
        lab = latest_observation(get, fhir_base, facts.mrn, code_for["potassium"])
        step, per_step, goal = facts.k_rule
        v.evidence["potassium"] = lab and {"value": lab["value"], "time": lab["time"].isoformat()}
        if lab is None:
            _block(v, "no potassium result on record")
            return
        if lab["value"] >= goal:
            _block(v, "latest potassium %.2f is not below the goal of %g" % (lab["value"], goal))
            return
        if endpoint == "MedicationRequest" and dose is not None:
            want = round((goal - lab["value"]) / step) * per_step
            if dose.get("value") != want or str(dose.get("unit", "")).lower() != "meq":
                v.issues.append("Dose must be %g mEq for potassium %.2f (task protocol)." % (want, lab["value"]))
    elif facts.max_age_days and endpoint == "ServiceRequest" and code_for:
        code = next(iter(code_for.values()))
        lab = latest_observation(get, fhir_base, facts.mrn, code)
        v.evidence["last_result"] = lab and {"value": lab["value"], "time": lab["time"].isoformat()}
        if lab and facts.now and (facts.now - lab["time"]).days <= facts.max_age_days:
            _block(v, "the last result (%s) is not older than %g days" % (lab["time"].date(), facts.max_age_days))


def _block(v: Verdict, reason: str) -> None:
    v.ok = False
    v.not_indicated = True
    v.issues.append("No order is indicated: %s." % reason)
