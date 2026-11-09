"""Base-paper protocol fidelity: the runner must behave exactly like upstream start_sample."""
import json
import os

import pytest

from medguardian import mab, runner
from medguardian.llm import ChatResult, ContextLimitError
from medguardian.tracing import Tracer, read_trace

CASE = {"id": "task1_1", "instruction": "What is the MRN of Peter Stafford, DOB 1932-12-29?",
        "context": "", "sol": ["S6534835"], "eval_MRN": "S6534835"}
CFG = {"fhir_base": "http://localhost:8080/fhir/", "max_rounds": 8}


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, prefix_tokens=0):
        self.calls.append((messages, prefix_tokens))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(content=reply, tokens_in=100 * len(messages), tokens_out=10, latency_ms=5.0)


class FakeFhir:
    def __init__(self, data='{"resourceType": "Bundle"}', error=None):
        self.urls = []
        self.data, self.error = data, error

    def get(self, url):
        self.urls.append(url)
        return ({"error": self.error} if self.error else {"status_code": 200, "data": self.data}), False


@pytest.fixture
def no_grader(monkeypatch):
    seen = {}

    def fake_grade(case, result, history, fhir_base):
        seen.update(result=result, history=list(history))
        return result == '["S6534835"]', "", None

    monkeypatch.setattr(mab, "grade", fake_grade)
    return seen


def _run(tmp_path, replies, fhir=None, cfg=CFG):
    llm = ScriptedLLM(replies)
    fhir = fhir or FakeFhir()
    with Tracer(str(tmp_path / "t.jsonl"), run_id="test") as tr:
        row = runner.run_task(CASE, [], llm, fhir, cfg, tr)
    return row, llm, fhir, read_trace(str(tmp_path / "t.jsonl"))


# -- action parsing --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("GET http://x/fhir/Patient?name=A", ("GET", "http://x/fhir/Patient?name=A&_format=json")),
    ("```tool_code\nGET http://x/fhir/Patient?a=1\n```", ("GET", "http://x/fhir/Patient?a=1&_format=json")),
    ("GET http://x/fhir/metadata", ("GET", "http://x/fhir/metadata&_format=json")),  # upstream quirk kept
    ('POST http://x/fhir/Observation\n{"a": 1}', ("POST", {"a": 1})),
    ("POST http://x/fhir/Observation\nnot json", ("POST", None)),
    ('FINISH(["S6534835"])', ("FINISH", '["S6534835"]')),
    ("FINISH([])", ("FINISH", "[]")),
    ("The MRN is S6534835", ("INVALID", None)),
    ("", ("INVALID", None)),
])
def test_parse_action_matches_upstream(raw, expected):
    assert runner.parse_action(raw) == expected


# -- loop semantics --------------------------------------------------------------------------
def test_get_then_finish_is_completed_and_graded(tmp_path, no_grader):
    row, llm, fhir, trace = _run(tmp_path, ["GET http://localhost:8080/fhir/Patient?name=Peter",
                                            'FINISH(["S6534835"])'])
    assert row["status"] == runner.COMPLETED and row["correct"] == 1
    assert row["rounds"] == 2 and row["n_get"] == 1
    assert fhir.urls == ["http://localhost:8080/fhir/Patient?name=Peter&_format=json"]
    second_turn = llm.calls[1][0]
    assert [m["role"] for m in second_turn] == ["user", "assistant", "user"]
    assert second_turn[2]["content"].startswith("Here is the response from the GET request:\n")
    assert [r for r, _ in no_grader["history"]] == ["user", "agent", "user", "agent"]
    assert {e["event"] for e in trace} >= {"task_start", "step", "task_end"}


def test_get_error_is_reported_to_agent(tmp_path, no_grader):
    _, llm, _, _ = _run(tmp_path, ["GET http://x?a=1", "FINISH([])"], fhir=FakeFhir(error="boom"))
    assert llm.calls[1][0][-1]["content"] == "Error in sending the GET request: boom"


def test_invalid_post_does_not_end_the_task(tmp_path, no_grader):
    row, llm, _, _ = _run(tmp_path, ["POST http://x/Observation\n{bad", 'POST http://x/Observation\n{"ok": 1}',
                                     "FINISH([])"])
    assert row["status"] == runner.COMPLETED and row["n_post"] == 2
    assert llm.calls[1][0][-1]["content"] == runner.POST_BAD
    assert llm.calls[2][0][-1]["content"] == runner.POST_OK


def test_free_text_is_invalid_action_and_not_graded(tmp_path, no_grader):
    row, _, _, _ = _run(tmp_path, ["I think the answer is S6534835"])
    assert row["status"] == runner.INVALID_ACTION and row["correct"] == 0
    assert no_grader["result"] is None


def test_round_cap_gives_task_limit_reached(tmp_path, no_grader):
    row, llm, _, _ = _run(tmp_path, ["GET http://x?a=1"] * 3, cfg=dict(CFG, max_rounds=3))
    assert row["status"] == runner.LIMIT_REACHED and row["rounds"] == 3 and len(llm.calls) == 3
    assert no_grader["result"] is None


def test_context_overflow_is_recorded_not_truncated(tmp_path, no_grader):
    row, _, _, _ = _run(tmp_path, ["GET http://x?a=1", ContextLimitError("too long")])
    assert row["status"] == runner.CONTEXT_LIMIT and row["correct"] == 0


def test_prefix_tokens_grow_between_rounds(tmp_path, no_grader):
    _, llm, _, _ = _run(tmp_path, ["GET http://x?a=1", "GET http://x?b=2", "FINISH([])"])
    prefixes = [p for _, p in llm.calls]
    assert prefixes[0] == 0 and prefixes[1] > 0 and prefixes[2] > prefixes[1]


# -- upstream assets ------------------------------------------------------------------------
def test_prompt_is_upstreams_verbatim():
    p = mab.upstream_prompt()
    assert p.startswith("You are an expert in using FHIR functions")
    assert "FINISH([answer1, answer2, ...])" in p and "{functions}" in p
    filled = mab.build_prompt(CASE, [{"name": "f"}], "http://h/fhir/")
    assert "Note that you should use http://h/fhir/ as the api_base." in filled
    assert filled.endswith("Question: " + CASE["instruction"])


def test_query_and_action_categories_partition_the_benchmark():
    assert set(mab.QUERY_CATEGORIES) | set(mab.ACTION_CATEGORIES) == set(mab.CATEGORIES)
    kinds = [mab.task_kind(c["id"]) for c in mab.load_cases()]
    assert kinds.count("query") == 150 and kinds.count("action") == 150


@pytest.mark.skipif(not os.path.exists(os.path.join(mab.TASK_PKG_DIR, "refsol.py")),
                    reason="refsol.py is distributed separately and not committed")
def test_real_grader_accepts_correct_task1_and_rejects_post():
    case = next(c for c in mab.load_cases() if c["id"] == "task1_1")
    good = json.dumps(case["sol"])
    assert mab.grade(case, good, [("user", "q"), ("agent", "FINISH(%s)" % good)], CFG["fhir_base"])[0] is True
    posted = [("user", "q"), ("agent", 'POST http://x\n{"a":1}'), ("user", runner.POST_OK)]
    assert mab.grade(case, good, posted, CFG["fhir_base"])[0] is False
    assert mab.grade(case, None, [], CFG["fhir_base"])[0] is False
