"""Renal dosing tools (Gap 1).

Pure functions (cockcroft_gault_crcl, renal_adjusted_dose) are kept separate from the
contract adapters at the bottom, which take a PatientContext and return a ToolCallRecord.

Phase 1 seed table: 5 drugs. Band values follow the FDA labels cited per drug and must be
re-checked against current labels before clinical use. This is a research prototype.
"""
import time
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict

from ..state import DosingRecommendation, PatientContext, ToolCallRecord


def _round1(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def cockcroft_gault_crcl(age_years: float, weight_kg: float, scr_mg_dl: float, sex: str) -> float:
    """Cockcroft-Gault creatinine clearance (mL/min) using total body weight, 1 decimal."""
    if scr_mg_dl is None or scr_mg_dl <= 0:
        raise ValueError("serum creatinine must be > 0 mg/dL")
    if age_years is None or not (18 <= age_years <= 120):
        raise ValueError("age must be between 18 and 120 years")
    if weight_kg is None or weight_kg <= 0:
        raise ValueError("weight must be > 0 kg")
    s = (sex or "").strip().lower()
    if s not in ("male", "female", "m", "f"):
        raise ValueError("sex must be male or female")
    crcl = (140 - age_years) * weight_kg / (72 * scr_mg_dl)
    if s in ("female", "f"):
        crcl *= 0.85
    return _round1(crcl)


# Each band: (label, predicate on CrCl, rule). Rule kinds:
#   ("fixed", mg)        -> dose is mg regardless of requested
#   ("cap", mg)          -> min(requested, mg)
#   ("halve_cap", mg)    -> min(requested / 2, mg)
#   ("standard", None)   -> requested dose unchanged
#   ("contra", None)     -> 0 mg, contraindicated
DRUG_TABLE: Dict[str, Dict[str, Any]] = {
    "metformin": {
        "basis": "total daily dose", "frequency": "per day (divided)",
        # Label uses eGFR; CrCl is used here as a surrogate.
        "source": "FDA metformin label (eGFR-based; CrCl used as surrogate) + KDIGO 2022 diabetes in CKD",
        "bands": [
            (">=45", lambda c: c >= 45, ("standard", None)),
            ("30-44", lambda c: 30 <= c < 45, ("halve_cap", 1000)),
            ("<30", lambda c: c < 30, ("contra", None)),
        ],
    },
    "enoxaparin": {
        "basis": "per dose (VTE prophylaxis)", "frequency": "once daily",
        "source": "Lovenox (enoxaparin) FDA label",
        "bands": [
            (">=30", lambda c: c >= 30, ("fixed", 40)),
            ("<30", lambda c: c < 30, ("fixed", 30)),
        ],
    },
    "gabapentin": {
        "basis": "total daily dose", "frequency": "per day (divided)",
        "source": "Neurontin (gabapentin) FDA label",
        "bands": [
            (">=60", lambda c: c >= 60, ("cap", 3600)),
            ("30-59", lambda c: 30 <= c < 60, ("cap", 1400)),
            ("15-29", lambda c: 15 <= c < 30, ("cap", 700)),
            ("<15", lambda c: c < 15, ("cap", 300)),
        ],
    },
    "levetiracetam": {
        "basis": "per dose", "frequency": "every 12 hours",
        # Label bands are in mL/min/1.73 m2 (BSA-normalised); unadjusted CG CrCl used in Phase 1.
        "source": "Keppra (levetiracetam) FDA label (bands are mL/min/1.73 m2)",
        "bands": [
            (">80", lambda c: c > 80, ("cap", 1500)),
            ("50-80", lambda c: 50 <= c <= 80, ("cap", 1000)),
            ("30-49", lambda c: 30 <= c < 50, ("cap", 750)),
            ("<30", lambda c: c < 30, ("cap", 500)),
        ],
    },
    "dabigatran": {
        "basis": "per dose (non-valvular AF)", "frequency": "twice daily",
        "source": "Pradaxa (dabigatran) FDA label",
        "bands": [
            (">30", lambda c: c > 30, ("fixed", 150)),
            ("15-30", lambda c: 15 <= c <= 30, ("fixed", 75)),
            ("<15", lambda c: c < 15, ("contra", None)),
        ],
    },
}


def renal_adjusted_dose(drug: str, crcl_ml_min: float, standard_dose_mg: float) -> Dict[str, Any]:
    """Look up the renal band for `drug` and return the adjusted dose. Unknown drug -> KeyError."""
    key = (drug or "").strip().lower()
    if key not in DRUG_TABLE:
        raise KeyError("no renal dosing rule for drug '%s' (supported: %s)"
                       % (drug, ", ".join(sorted(DRUG_TABLE))))
    if crcl_ml_min is None or crcl_ml_min < 0:
        raise ValueError("CrCl must be >= 0")
    entry = DRUG_TABLE[key]
    for label, pred, (kind, mg) in entry["bands"]:
        if not pred(crcl_ml_min):
            continue
        contraindicated = False
        if kind == "fixed":
            dose = float(mg)
        elif kind == "cap":
            dose = float(min(standard_dose_mg, mg))
        elif kind == "halve_cap":
            dose = float(min(standard_dose_mg / 2.0, mg))
        elif kind == "standard":
            dose = float(standard_dose_mg)
        else:
            dose, contraindicated = 0.0, True
        rationale = "CrCl %.1f mL/min falls in band %s (%s): " % (crcl_ml_min, label, entry["basis"])
        rationale += "contraindicated / not recommended" if contraindicated else "%s mg %s" % (dose, entry["frequency"])
        return {
            "drug": key, "crcl_ml_min": crcl_ml_min, "band": label,
            "adjusted_dose_mg": dose, "frequency": entry["frequency"],
            "contraindicated": contraindicated, "rationale": rationale, "source": entry["source"],
        }
    raise ValueError("CrCl %s matched no band for %s" % (crcl_ml_min, key))


# ---- contract adapters: (patient_context, **kwargs) -> ToolCallRecord ----

def crcl_tool(ctx: PatientContext, age_years: float = None, weight_kg: float = None,
              scr_mg_dl: float = None, sex: str = None) -> ToolCallRecord:
    args = {"age_years": age_years if age_years is not None else ctx.age_years,
            "weight_kg": weight_kg if weight_kg is not None else ctx.weight_kg,
            "scr_mg_dl": scr_mg_dl if scr_mg_dl is not None else ctx.serum_creatinine_mg_dl,
            "sex": sex or ctx.sex}
    t0 = time.time()
    try:
        crcl = cockcroft_gault_crcl(**args)
        ctx.crcl_ml_min = crcl
        rec = ToolCallRecord(tool_name="cockcroft_gault_crcl", arguments=args, output={"crcl_ml_min": crcl})
    except (ValueError, TypeError) as e:
        rec = ToolCallRecord(tool_name="cockcroft_gault_crcl", arguments=args, success=False, error=str(e))
    rec.latency_ms = (time.time() - t0) * 1000
    return ctx.record(rec)


def dose_tool(ctx: PatientContext, drug: str = None, standard_dose_mg: float = None,
              crcl_ml_min: float = None) -> ToolCallRecord:
    crcl = crcl_ml_min if crcl_ml_min is not None else ctx.crcl_ml_min
    args = {"drug": drug, "crcl_ml_min": crcl, "standard_dose_mg": standard_dose_mg}
    t0 = time.time()
    try:
        out = renal_adjusted_dose(drug, crcl, float(standard_dose_mg))
        ctx.recommendations.append(DosingRecommendation(**out))
        rec = ToolCallRecord(tool_name="renal_adjusted_dose", arguments=args, output=out)
    except (KeyError, ValueError, TypeError) as e:
        rec = ToolCallRecord(tool_name="renal_adjusted_dose", arguments=args, success=False, error=str(e))
    rec.latency_ms = (time.time() - t0) * 1000
    return ctx.record(rec)
