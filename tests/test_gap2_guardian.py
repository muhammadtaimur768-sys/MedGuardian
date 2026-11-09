"""Gap 2: guideline retrieval, timeline intervals, severity, generator determinism, E4-G2 metrics."""
import copy
import json
import os
import random
import sys

import pytest

from medguardian.guardian import severity
from medguardian.guardian.process_guardian import audit, build_timeline, patient_crcl
from medguardian.tools.guideline_rag import default_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data", "generators"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import timeline as gen  # noqa: E402
import run_e4_g2  # noqa: E402


def case_with(drug="apixaban", bleed="high", last_hold_h=60, restart_h=60, plt=200, inr=None, plt_age_h=24):
    res = [{"resourceType": "Patient", "id": "P1", "gender": "male", "birthDate": "1950-01-01"},
           {"resourceType": "Procedure", "category": {"text": "%s bleeding risk" % bleed}, "code": {"text": "x"},
            "performedPeriod": {"start": "2023-11-20T08:00:00+00:00", "end": "2023-11-20T10:00:00+00:00"}},
           {"resourceType": "Observation", "code": {"coding": [{"code": "CREAT"}]},
            "effectiveDateTime": "2023-11-19T08:00:00+00:00", "valueQuantity": {"value": 1.0}},
           {"resourceType": "Observation", "code": {"coding": [{"code": "WEIGHT"}]},
            "effectiveDateTime": "2023-11-19T08:00:00+00:00", "valueQuantity": {"value": 80}}]
    from datetime import datetime, timedelta
    start = datetime.fromisoformat("2023-11-20T08:00:00+00:00")
    end = start + timedelta(hours=2)
    for t in (start - timedelta(hours=last_hold_h), end + timedelta(hours=restart_h)):
        res.append({"resourceType": "MedicationAdministration", "medicationCodeableConcept": {"text": drug},
                    "effectiveDateTime": t.isoformat()})
    if plt is not None:
        res.append({"resourceType": "Observation", "code": {"coding": [{"code": "PLT"}]},
                    "effectiveDateTime": (start - timedelta(hours=plt_age_h)).isoformat(), "valueQuantity": {"value": plt}})
    if inr is not None:
        res.append({"resourceType": "Observation", "code": {"coding": [{"code": "INR"}]},
                    "effectiveDateTime": (start - timedelta(hours=12)).isoformat(), "valueQuantity": {"value": inr}})
    return {"entry": [{"resource": r} for r in res]}


def types(flags):
    return sorted((f["type"], f["item"]) for f in flags)


def test_clean_record_has_no_flags():
    assert audit(case_with()) == []


def test_hold_and_restart_violations_cite_their_guideline():
    flags = audit(case_with(last_hold_h=30, restart_h=20))
    assert types(flags) == [("hold_too_short", "apixaban"), ("restart_too_early", "apixaban")]
    assert all(f["citation"] and f["guideline_id"] and f["severity"] == "high" for f in flags)


def test_low_bleeding_risk_uses_the_shorter_interval():
    assert audit(case_with(bleed="low", last_hold_h=30, restart_h=30)) == []


def test_lab_rules_out_of_range_and_missing():
    assert types(audit(case_with(plt=40))) == [("preop_lab_out_of_range", "PLT")]
    assert types(audit(case_with(plt_age_h=200))) == [("missing_required_lab", "PLT")]
    assert ("missing_required_lab", "INR") in types(audit(case_with(drug="warfarin", last_hold_h=130)))
    assert ("preop_lab_out_of_range", "INR") in types(audit(case_with(drug="warfarin", last_hold_h=130, inr=1.8)))


def test_dabigatran_interval_depends_on_crcl():
    ix = default_index()
    assert ix.hold_rule("dabigatran", 70)["facts"]["min_hold_hours"]["high"] == 48
    assert ix.hold_rule("dabigatran", 30)["facts"]["min_hold_hours"]["high"] == 96


def test_crcl_is_computed_from_the_record():
    tl = build_timeline(case_with())
    assert patient_crcl(tl) == pytest.approx((140 - 73) * 80 / 72, abs=0.1)


def test_severity_rules():
    assert severity.severity("hold_too_short", "high") == "high"
    assert severity.severity("hold_too_short", "low") == "medium"
    assert severity.severity("missing_required_lab", "high") == "medium"


def test_generator_is_deterministic_and_balanced():
    a = [gen.make_case(i, random.Random(0), "test") for i in range(1)]
    rng1, rng2 = random.Random(0), random.Random(0)
    assert [gen.make_case(i, rng1, "t") for i in range(5)] == [gen.make_case(i, rng2, "t") for i in range(5)]
    cases = [json.loads(l) for l in open(os.path.join(ROOT, "data", "gap2", "test.jsonl"))]
    n_viol = sum(bool(c["gold"]) for c in cases)
    assert 0.25 * len(cases) <= n_viol <= 0.75 * len(cases)


def test_metrics():
    cases = [{"id": "a", "gold": [{"type": "hold_too_short", "item": "apixaban", "severity": "high"}]},
             {"id": "b", "gold": []}]
    s = run_e4_g2.score(cases, {"a": [], "b": [{"type": "missing_required_lab", "item": "plt"}]})
    assert (s["tp"], s["fp"], s["fn"], s["f1"]) == (0, 1, 1, 0.0)
    assert s["false_alert_rate"] == 1.0 and s["severity_weighted_miss_rate"] == 1.0


def test_llm_output_parsing_is_tolerant():
    flags, ok = run_e4_g2.parse_flags('Here: [{"type": "hold_too_short", "item": "Apixaban"}, {"type": "bogus"}]')
    assert ok and flags == [{"type": "hold_too_short", "item": "apixaban"}]
    assert run_e4_g2.parse_flags("no json")[1] is False
