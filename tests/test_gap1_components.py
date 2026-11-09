"""Gap 1 components: task-text parsing, typed FHIR tools, calculator, order verifier, MedGuardian loop."""
import json
from datetime import datetime, timezone

import pytest

from medguardian import mab, runner
from medguardian.llm import ChatResult
from medguardian.tools import fhir_summary as fs
from medguardian.tools import task_context as tc
from medguardian.tracing import Tracer
from medguardian.verify import order_verifier as ov

NOW = "2023-11-13T10:15:00+00:00"
MG_CASE = {
    "id": "task5_99",
    "instruction": "Check patient S1234567's last serum magnesium level within last 24 hours. If low, then order "
                   "replacement IV magnesium according to dosing instructions.",
    "context": "It's 2023-11-13T10:15:00+00:00 now. The code for magnesium is \"MG\". The NDC for replacement IV "
               "magnesium is 0338-1715-40. Dosing instructions: (1) Mild deficiency (eg, serum magnesium 1.5 to 1.9 "
               "mg/dL):\nIV: 1 g over 1 hour.\n(2) Moderate deficiency (eg, serum magnesium 1 to <1.5 mg/dL):\nIV: 2 g "
               "over 2 hours.\n(3) Severe deficiency (eg, serum magnesium <1 mg/dL):\nIV: 4 g over 4 hours.\n",
}
QUERY_CASE = {"id": "task1_99", "instruction": "What is the MRN of Jane Roe, DOB 1950-01-01?", "context": "", "sol": ["S1"]}


def bundle(*resources):
    return {"resourceType": "Bundle", "entry": [{"resource": r} for r in resources], "link": []}


def obs(value, when, code="MG"):
    return {"resourceType": "Observation", "id": when, "code": {"coding": [{"code": code}]},
            "effectiveDateTime": when, "valueQuantity": {"value": value, "unit": "mg/dL"}}


def fake_get(resources):
    def get(url):
        return {"status_code": 200, "data": json.dumps(bundle(*resources))}, False
    return get


def mg_order(grams=2, hours=2, mrn="S1234567", ndc="0338-1715-40"):
    return {"resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": ndc}]},
            "authoredOn": NOW, "status": "active", "intent": "order", "subject": {"reference": "Patient/" + mrn},
            "dosageInstruction": [{"route": {"text": "IV"}, "doseAndRate": [
                {"doseQuantity": {"value": grams, "unit": "g"}, "rateQuantity": {"value": hours, "unit": "h"}}]}]}


# -- task text ------------------------------------------------------------------------------
def test_parse_reads_every_fact_from_the_task_text():
    f = tc.parse(MG_CASE)
    assert f.now == datetime(2023, 11, 13, 10, 15, tzinfo=timezone.utc)
    assert (f.mrn, f.ndc, f.window_hours, f.lab_codes) == ("S1234567", "0338-1715-40", 24.0, {"magnesium": "MG"})
    assert f.mg_tiers[0][:3] == (1.5, 1.9, True) and f.mg_tiers[1][:3] == (1.0, 1.5, False)


def test_parser_covers_all_300_benchmark_tasks():
    for c in mab.load_cases():
        f = tc.parse(c)
        cat = mab.category(c["id"])
        if cat != "task1":
            assert f.now and f.mrn, c["id"]
        if cat == "task5":
            assert len(f.mg_tiers) == 3, c["id"]


@pytest.mark.parametrize("value,expected", [(1.9, (1.0, 1.0)), (1.5, (1.0, 1.0)), (1.49, (2.0, 2.0)),
                                            (1.0, (2.0, 2.0)), (0.99, (4.0, 4.0)), (2.0, None)])
def test_magnesium_tiers_follow_the_written_boundaries(value, expected):
    assert ov._mg_dose(tc.parse(MG_CASE).mg_tiers, value) == expected


# -- typed tools / calculator ---------------------------------------------------------------
def test_normalise_rewrites_date_ranges_and_dedupes_format():
    url, notes = fs.normalise_url("http://h/fhir/Observation?code=MG&date=2023-11-12T10:15:00+00:00/"
                                  "2023-11-13T10:15:00+00:00&_format=json&_format=json")
    assert "date=ge2023-11-12T10:15:00%2B00:00&date=le2023-11-13T10:15:00%2B00:00" in url
    assert url.count("_format=json") == 1 and notes


def test_summary_sorts_newest_first_and_computes_window_stats():
    f = tc.parse(MG_CASE)
    b = bundle(obs(2.0, "2023-11-12T08:00:00+00:00"), obs(1.4, "2023-11-13T09:15:00+00:00"),
               obs(1.8, "2023-11-12T20:15:00+00:00"))
    s = fs.summarise(b, f, use_calculator=True)
    assert [r["value"] for r in s["resources_newest_first"]] == [1.4, 1.8, 2.0]
    calc = s["calculator"]
    assert calc["latest_in_window"] == 1.4 and calc["count_in_window"] == 2 and calc["mean_in_window"] == 1.6
    assert "calculator" not in fs.summarise(b, f, use_calculator=False)


def test_patient_age_is_computed_as_of_task_time():
    f = tc.parse(MG_CASE)
    p = {"resourceType": "Patient", "id": "S1", "birthDate": "1939-11-14", "identifier": [{"value": "S1"}],
         "name": [{"given": ["A"], "family": "B"}]}
    row = fs.summarise(bundle(p), f, True)["resources_newest_first"][0]
    assert row["age_years_now"] == 83  # birthday is tomorrow


def test_fetch_all_follows_next_pages():
    pages = {"u1": {"entry": [{"resource": obs(1, "a")}], "link": [{"relation": "next", "url": "u2"}]},
             "u2&_format=json": {"entry": [{"resource": obs(2, "b")}], "link": []}}
    b, err = fs.fetch_all(lambda u: ({"data": pages[u]}, False), "u1")
    assert err is None and len(b["entry"]) == 2


# -- order verifier -------------------------------------------------------------------------
def test_correct_order_for_moderate_deficiency_is_accepted():
    v = ov.verify("http://h/fhir/MedicationRequest", mg_order(2, 2), tc.parse(MG_CASE),
                  fake_get([obs(1.2, "2023-11-13T06:00:00+00:00")]), "http://h/fhir/")
    assert v.ok, v.issues


def test_wrong_dose_patient_and_code_are_reported():
    v = ov.verify("http://h/fhir/MedicationRequest", mg_order(4, 4, mrn="S0", ndc="0"), tc.parse(MG_CASE),
                  fake_get([obs(1.2, "2023-11-13T06:00:00+00:00")]), "http://h/fhir/")
    text = " ".join(v.issues)
    assert not v.ok and not v.not_indicated
    assert "Patient/S1234567" in text and "0338-1715-40" in text and "2 g" in text


def test_order_without_recent_lab_is_not_indicated():
    v = ov.verify("http://h/fhir/MedicationRequest", mg_order(), tc.parse(MG_CASE),
                  fake_get([obs(1.2, "2023-11-10T06:00:00+00:00")]), "http://h/fhir/")
    assert v.not_indicated and not v.ok


def test_wrong_endpoint_and_invalid_json_are_rejected():
    f = tc.parse(MG_CASE)
    assert not ov.verify("http://h/fhir/", mg_order(), f, fake_get([]), "http://h/fhir/").ok
    assert not ov.verify("http://h/fhir/MedicationRequest", None, f, fake_get([]), "http://h/fhir/").ok


# -- MedGuardian loop -----------------------------------------------------------------------
class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, prefix_tokens=0):
        self.calls.append(messages)
        return ChatResult(content=self.replies.pop(0), tokens_in=100 * len(messages), tokens_out=10, latency_ms=1.0)


class Fhir:
    def __init__(self, resources):
        self.get = fake_get(resources)


MG_CFG = {"fhir_base": "http://h/fhir/", "max_rounds": 8, "mode": "medguardian",
          "medguardian": {"use_typed_tools": True, "use_calculator": True, "use_order_verifier": True}}


@pytest.fixture
def graded(monkeypatch):
    seen = {}

    def fake_grade(case, result, history, fhir_base):
        seen["history"] = list(history)
        return result is not None, "", None

    monkeypatch.setattr(mab, "grade", fake_grade)
    return seen


def run(tmp_path, case, replies, resources, cfg=MG_CFG):
    llm = ScriptedLLM(replies)
    with Tracer(str(tmp_path / "t.jsonl"), run_id="t") as tr:
        row = runner.run_task(case, [], llm, Fhir(resources), cfg, tr)
    return row, llm


def post(order):
    return "POST http://h/fhir/MedicationRequest\n" + json.dumps(order)


def test_rejected_draft_is_repaired_and_only_the_fix_reaches_the_ehr(tmp_path, graded):
    lab = [obs(1.2, "2023-11-13T06:00:00+00:00")]
    row, llm = run(tmp_path, MG_CASE, [post(mg_order(4, 4)), post(mg_order(2, 2)), "FINISH([1.2])"], lab)
    assert (row["n_post"], row["n_post_rejected"], row["n_post_blocked"]) == (1, 1, 0)
    assert "NOT sent" in llm.calls[1][-1]["content"]
    env_agent = [c for r, c in graded["history"] if r == "agent"]
    assert env_agent == [post(mg_order(2, 2)), "FINISH([1.2])"]


def test_second_failure_blocks_the_order(tmp_path, graded):
    lab = [obs(1.2, "2023-11-13T06:00:00+00:00")]
    row, _ = run(tmp_path, MG_CASE, [post(mg_order(4, 4)), post(mg_order(4, 4)), "FINISH([])"], lab)
    assert (row["n_post"], row["n_post_rejected"], row["n_post_blocked"]) == (0, 2, 1)
    assert not any("POST" in c for r, c in graded["history"] if r == "agent")


def test_free_text_gets_one_format_repair_and_wrapped_actions_are_recovered(tmp_path, graded):
    row, _ = run(tmp_path, QUERY_CASE, ["Let me think.", "Sure!\nFINISH([\"S1\"])"], [])
    assert row["status"] == runner.COMPLETED and row["n_format_repairs"] == 1
    assert graded["history"][-1] == ("agent", 'FINISH(["S1"])')


def test_all_flags_off_behaves_like_basepaper(tmp_path, graded):
    off = dict(MG_CFG, medguardian={"use_typed_tools": False, "use_calculator": False, "use_order_verifier": False})
    row, _ = run(tmp_path, QUERY_CASE, ["Let me think."], [], cfg=off)
    assert row["status"] == runner.INVALID_ACTION and row["n_format_repairs"] == 0
