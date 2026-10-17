import json
import os

import pytest

from medguardian.state import PatientContext
from medguardian.tools.dosing import cockcroft_gault_crcl, crcl_tool, dose_tool, renal_adjusted_dose

CASES = json.load(open(os.path.join(os.path.dirname(__file__), "..", "eval", "data", "seed_cases.json")))


def test_corrected_worked_example():
    # 72 y, 68 kg, female, SCr 1.4 -> 39.0 mL/min (brief 9.3 said 30.9, which is wrong)
    assert cockcroft_gault_crcl(72, 68, 1.4, "female") == 39.0


def test_male_formula():
    assert cockcroft_gault_crcl(80, 60, 2.0, "male") == 25.0


@pytest.mark.parametrize("args", [(72, 68, 0, "female"), (72, 68, -1, "male"), (17, 68, 1.0, "male"),
                                  (121, 68, 1.0, "male"), (72, 0, 1.0, "male"), (72, 68, 1.0, "x")])
def test_invalid_inputs_rejected(args):
    with pytest.raises(ValueError):
        cockcroft_gault_crcl(*args)


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_seed_cases_match_hand_computed(case):
    p, exp = case["patient"], case["expected"]
    age = 2023 - int(p["birthDate"][:4])  # all seed birthdays fall before 2023-11-13
    crcl = cockcroft_gault_crcl(age, p["weight_kg"], p["creat_mg_dl"], p["gender"])
    assert crcl == exp["crcl_ml_min"]
    out = renal_adjusted_dose(case["drug"], crcl, case["standard_dose_mg"])
    assert out["band"] == exp["band"]
    assert out["adjusted_dose_mg"] == exp["adjusted_dose_mg"]
    assert out["contraindicated"] == exp["contraindicated"]


@pytest.mark.parametrize("drug,crcl,std,band,dose", [
    ("metformin", 45.0, 2000, ">=45", 2000), ("metformin", 44.9, 2000, "30-44", 1000),
    ("metformin", 30.0, 1500, "30-44", 750), ("metformin", 29.9, 2000, "<30", 0),
    ("enoxaparin", 30.0, 40, ">=30", 40), ("enoxaparin", 29.9, 40, "<30", 30),
    ("gabapentin", 60.0, 3600, ">=60", 3600), ("gabapentin", 59.9, 3600, "30-59", 1400),
    ("gabapentin", 15.0, 3600, "15-29", 700), ("gabapentin", 14.9, 3600, "<15", 300),
    ("levetiracetam", 80.0, 1500, "50-80", 1000), ("levetiracetam", 80.1, 1500, ">80", 1500),
    ("levetiracetam", 50.0, 1500, "50-80", 1000), ("levetiracetam", 49.9, 1500, "30-49", 750),
    ("dabigatran", 30.0, 150, "15-30", 75), ("dabigatran", 30.1, 150, ">30", 150),
    ("dabigatran", 14.9, 150, "<15", 0),
])
def test_band_edges(drug, crcl, std, band, dose):
    out = renal_adjusted_dose(drug, crcl, std)
    assert (out["band"], out["adjusted_dose_mg"]) == (band, dose)


def test_contraindicated_flags():
    assert renal_adjusted_dose("dabigatran", 10, 150)["contraindicated"] is True
    assert renal_adjusted_dose("metformin", 20, 2000)["contraindicated"] is True


def test_unknown_drug_raises():
    with pytest.raises(KeyError):
        renal_adjusted_dose("warfarin", 50, 5)


def test_contract_adapters_record_calls():
    ctx = PatientContext(age_years=72, weight_kg=68, serum_creatinine_mg_dl=1.4, sex="female")
    rec = crcl_tool(ctx)
    assert rec.success and ctx.crcl_ml_min == 39.0
    rec = dose_tool(ctx, drug="metformin", standard_dose_mg=2000)
    assert rec.success and ctx.recommendations[0].adjusted_dose_mg == 1000
    bad = dose_tool(ctx, drug="warfarin", standard_dose_mg=5)
    assert not bad.success and "warfarin" in bad.error
    assert [c.tool_name for c in ctx.tool_calls] == ["cockcroft_gault_crcl", "renal_adjusted_dose", "renal_adjusted_dose"]
