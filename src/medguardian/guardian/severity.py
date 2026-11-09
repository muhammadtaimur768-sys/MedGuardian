"""Rule-based severity labels for process discrepancies (Gap 2).

Documented rules (no clinician review was available, declared in Limitations):
  * an interval violation (hold too short / restart too early) is HIGH around a high bleeding
    risk procedure and MEDIUM around a low bleeding risk one;
  * a pre-operative lab out of range is HIGH (it changes whether surgery can proceed);
  * a missing required lab is MEDIUM (the risk is unknown rather than known to be present).
Weights for the severity-weighted miss rate: high 3, medium 2, low 1.
"""
WEIGHTS = {"high": 3, "medium": 2, "low": 1}


def severity(kind: str, bleeding_risk: str) -> str:
    if kind in ("hold_too_short", "restart_too_early"):
        return "high" if bleeding_risk == "high" else "medium"
    if kind == "preop_lab_out_of_range":
        return "high"
    return "medium"
