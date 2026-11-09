"""Gap 2 process guardian: medication/lab timeline verifier with retrieved guideline thresholds.

audit(bundle) builds a time-ordered timeline from FHIR resources, then checks it:
  1. hold_too_short          last pre-procedure dose is closer to the start than the retrieved
                             minimum interruption (drug-, bleeding-risk- and CrCl-specific);
  2. restart_too_early       first post-procedure dose of a DOAC comes sooner than the retrieved
                             minimum resumption interval after the procedure ends;
  3. preop_lab_out_of_range  latest required lab inside its window breaks the retrieved threshold;
  4. missing_required_lab    no required lab inside the retrieved window.
Every threshold is read from the retrieved guideline chunk, and every discrepancy carries that
chunk's citation, its severity and the timeline evidence.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..tools.dosing import cockcroft_gault_crcl
from ..tools.guideline_rag import GuidelineIndex, default_index
from .severity import severity


@dataclass
class Discrepancy:
    type: str
    item: str
    severity: str
    evidence: str
    citation: str
    guideline_id: str


@dataclass
class Timeline:
    mrn: Optional[str] = None
    sex: Optional[str] = None
    birth: Optional[datetime] = None
    procedure: Optional[str] = None
    bleeding_risk: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    doses: Dict[str, List[datetime]] = field(default_factory=dict)
    labs: Dict[str, List[tuple]] = field(default_factory=dict)     # code -> [(time, value)]


def _t(s: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
    except ValueError:
        return None


def build_timeline(bundle: Dict[str, Any], index: Optional[GuidelineIndex] = None) -> Timeline:
    index = index or default_index()
    tl = Timeline()
    for e in bundle.get("entry", []):
        r = e.get("resource", {})
        rt = r.get("resourceType")
        if rt == "Patient":
            tl.mrn, tl.sex = r.get("id"), r.get("gender")
            tl.birth = _t(r.get("birthDate"))
        elif rt in ("MedicationAdministration", "MedicationRequest"):
            raw = ((r.get("medicationCodeableConcept") or {}).get("text") or "").strip()
            drug = index.resolve_drug(raw) or raw.lower()  # brand / free text -> generic via retrieval
            when = _t(r.get("effectiveDateTime") or r.get("authoredOn"))
            if drug and when:
                tl.doses.setdefault(drug, []).append(when)
        elif rt == "Observation":
            code = ((r.get("code") or {}).get("coding") or [{}])[0].get("code") or (r.get("code") or {}).get("text")
            when, val = _t(r.get("effectiveDateTime")), (r.get("valueQuantity") or {}).get("value")
            if code and when and isinstance(val, (int, float)):
                tl.labs.setdefault(code, []).append((when, val))
        elif rt == "Procedure":
            tl.procedure = (r.get("code") or {}).get("text")
            cat = ((r.get("category") or {}).get("text") or "").lower()
            if "high" in cat or "low" in cat:
                tl.bleeding_risk = "high" if "high" in cat else "low"
            else:  # no label in the record: classify the procedure from the retrieved guideline chunk
                br = index.bleeding_risk(tl.procedure or "")
                tl.bleeding_risk = br["level"] if br else "high"
            period = r.get("performedPeriod") or {}
            tl.start, tl.end = _t(period.get("start")), _t(period.get("end"))
    for v in tl.doses.values():
        v.sort()
    for v in tl.labs.values():
        v.sort()
    return tl


def _latest_before(series: List[tuple], t: datetime) -> Optional[tuple]:
    prior = [x for x in series if x[0] <= t]
    return prior[-1] if prior else None


def patient_crcl(tl: Timeline) -> Optional[float]:
    scr, wt = _latest_before(tl.labs.get("CREAT", []), tl.start), _latest_before(tl.labs.get("WEIGHT", []), tl.start)
    if not (scr and wt and tl.birth and tl.sex and tl.start):
        return None
    age = tl.start.year - tl.birth.year - ((tl.start.month, tl.start.day) < (tl.birth.month, tl.birth.day))
    try:
        return cockcroft_gault_crcl(age, wt[1], scr[1], tl.sex)
    except ValueError:
        return None


def _h(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600


def audit(bundle: Dict[str, Any], index: Optional[GuidelineIndex] = None) -> List[Dict[str, Any]]:
    index = index or default_index()
    tl = build_timeline(bundle, index)
    out: List[Discrepancy] = []
    if not tl.start:
        return []
    cl = patient_crcl(tl)
    risk = tl.bleeding_risk or "high"
    for drug, times in tl.doses.items():
        rule = index.hold_rule(drug, cl)
        if rule:
            need = rule["facts"]["min_hold_hours"][risk]
            last = _latest_before(times and [(t, None) for t in times], tl.start)
            if last and _h(last[0], tl.start) < need:
                out.append(Discrepancy("hold_too_short", drug, severity("hold_too_short", risk),
                                       "last dose %s, %.1f h before the procedure; at least %g h required (%s bleeding risk%s)"
                                       % (last[0].isoformat(), _h(last[0], tl.start), need, risk,
                                          ", CrCl %.0f mL/min" % cl if cl is not None and drug == "dabigatran" else ""),
                                       rule["cite"], rule["id"]))
        resume = index.resume_rule(drug)
        if resume and tl.end:
            need = resume["facts"]["min_resume_hours"][risk]
            after = [t for t in times if t > tl.start]
            if after and _h(tl.end, after[0]) < need:
                out.append(Discrepancy("restart_too_early", drug, severity("restart_too_early", risk),
                                       "restarted %s, %.1f h after the procedure ended; at least %g h required"
                                       % (after[0].isoformat(), _h(tl.end, after[0]), need), resume["cite"], resume["id"]))
    for rule in index.lab_rules(list(tl.doses)):
        f = rule["facts"]
        lab = f["lab"]
        inside = [x for x in tl.labs.get(lab, []) if 0 <= _h(x[0], tl.start) <= f["within_hours"]]
        if not inside:
            out.append(Discrepancy("missing_required_lab", lab, severity("missing_required_lab", risk),
                                   "no %s within %g h before the procedure" % (lab, f["within_hours"]),
                                   rule["cite"], rule["id"]))
            continue
        when, value = inside[-1]
        if ("max_value" in f and value > f["max_value"]) or ("min_value" in f and value < f["min_value"]):
            limit = "<= %g" % f["max_value"] if "max_value" in f else ">= %g" % f["min_value"]
            out.append(Discrepancy("preop_lab_out_of_range", lab, severity("preop_lab_out_of_range", risk),
                                   "%s %g at %s; required %s" % (lab, value, when.isoformat(), limit),
                                   rule["cite"], rule["id"]))
    return [asdict(d) for d in out]
