"""Orchestrator loop tests with a scripted fake LLM (tests the loop only; not model results)."""
from medguardian.llm_client import LLMResponse
from medguardian.orchestrator import Orchestrator, human_approval_gate


def _call(i, name, **args):
    import json
    return {"id": "c%d" % i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedLLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools=None, temperature=0.0):
        self.seen.append(list(messages))
        calls = self.turns.pop(0)
        msg = {"role": "assistant", "content": None, "tool_calls": calls} if calls else {"role": "assistant", "content": "plain"}
        return LLMResponse(msg, {"prompt_tokens": 10, "completion_tokens": 5}, 1.0)


def test_full_loop_against_mock_server(mock_fhir):
    llm = ScriptedLLM([
        [_call(1, "query_patient", given="Amina", family="Rauf", birthdate="1951-01-15")],
        [_call(2, "query_observation", code="CREAT"), _call(3, "query_observation", code="WEIGHT")],
        [_call(4, "cockcroft_gault_crcl", age_years=72, weight_kg=68, scr_mg_dl=1.4, sex="female")],
        [_call(5, "renal_adjusted_dose", drug="metformin", crcl_ml_min=39.0, standard_dose_mg=2000)],
        [_call(6, "finish", answer="CrCl 39.0; metformin 1000 mg/day")],
    ])
    res = Orchestrator(llm, mock_fhir).run("dose metformin for Amina Rauf")
    ctx = res["context"]
    assert res["status"] == "finished"
    assert [c.tool_name for c in ctx.tool_calls] == ["query_patient", "query_observation", "query_observation",
                                                     "cockcroft_gault_crcl", "renal_adjusted_dose"]
    assert ctx.recommendations[0].adjusted_dose_mg == 1000
    assert res["usage"]["llm_calls"] == 5 and res["usage"]["prompt_tokens"] == 50
    # tool results are fed back to the model
    assert any(m.get("role") == "tool" for m in llm.seen[-1])


def test_bad_arguments_and_unknown_tool_do_not_crash(mock_fhir):
    llm = ScriptedLLM([[_call(1, "cockcroft_gault_crcl", bogus=1), _call(2, "delete_patient")], None])
    res = Orchestrator(llm, mock_fhir).run("x")
    errs = [c for c in res["context"].tool_calls if not c.success]
    assert len(errs) == 2 and res["status"] == "finished_text"


def test_max_rounds():
    llm = ScriptedLLM([[_call(i, "cockcroft_gault_crcl", age_years=50, weight_kg=70, scr_mg_dl=1, sex="male")] for i in range(3)])
    res = Orchestrator(llm, "http://127.0.0.1:1/fhir/", max_rounds=3).run("loop")
    assert res["status"] == "max_rounds"


def test_approval_gate():
    from medguardian.state import DosingRecommendation, PatientContext
    ctx = PatientContext()
    assert human_approval_gate(ctx, input_fn=lambda _: "y") is False  # nothing to approve
    ctx.recommendations.append(DosingRecommendation(drug="m", crcl_ml_min=1, band="b", adjusted_dose_mg=0,
                                                    frequency="f", contraindicated=True, rationale="r", source="s"))
    assert human_approval_gate(ctx, input_fn=lambda _: "y") is True
    assert human_approval_gate(ctx, input_fn=lambda _: "") is False
