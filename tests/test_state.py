import json

from medguardian.state import DosingRecommendation, PatientContext, ToolCallRecord, to_dict


def test_timestamps_not_shared():
    a, b = ToolCallRecord(tool_name="x"), ToolCallRecord(tool_name="y")
    assert a.call_id != b.call_id
    assert a.timestamp and b.timestamp


def test_mutable_defaults_independent():
    a, b = PatientContext(), PatientContext()
    a.record(ToolCallRecord(tool_name="t"))
    a.labs["CREAT"] = 1.0
    assert b.tool_calls == [] and b.labs == {}


def test_to_dict_is_json_serialisable():
    ctx = PatientContext(patient_id="MG1001", age_years=72)
    ctx.recommendations.append(DosingRecommendation(drug="metformin", crcl_ml_min=39.0, band="30-44",
                                                    adjusted_dose_mg=1000, frequency="per day", contraindicated=False,
                                                    rationale="r", source="s"))
    d = to_dict(ctx)
    assert json.loads(json.dumps(d))["recommendations"][0]["adjusted_dose_mg"] == 1000
    assert d["requires_human_approval"] is True


def test_trace_is_human_readable():
    ctx = PatientContext(patient_id="MG1001")
    ctx.record(ToolCallRecord(tool_name="query_patient", arguments={"given": "Amina"}, output=[{"id": "MG1001"}], latency_ms=3))
    t = ctx.trace()
    assert "query_patient" in t and "MG1001" in t
