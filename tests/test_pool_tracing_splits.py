"""LLM cache + overflow detection, JSONL tracing, and split determinism."""
import json
import os
import sys

import pytest

from medguardian.llm import ChatModel, ContextLimitError, LLMError
from medguardian.llm.pool import cache_key, min_prompt_tokens
from medguardian.tracing import Tracer, read_trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
import make_splits  # noqa: E402


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if isinstance(body, dict) else body

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def post(self, url, json=None, timeout=None):
        self.bodies.append(json)
        return self.responses.pop(0)


def ok(content="FINISH([])", prompt_tokens=50, completion_tokens=5):
    return FakeResponse(200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})


MSGS = [{"role": "user", "content": "hello"}]


def model(tmp_path, responses, **kw):
    return ChatModel("m", cache_dir=str(tmp_path / "cache"), session=FakeSession(responses), **kw)


def test_cache_hit_skips_the_server(tmp_path):
    m = model(tmp_path, [ok("GET x")])
    first = m.chat(MSGS)
    second = m.chat(MSGS)
    assert first.content == second.content == "GET x"
    assert not first.cached and second.cached
    assert len(m.http.bodies) == 1


def test_cache_key_depends_on_seed_and_temperature():
    a = cache_key("m", MSGS, 0.0, 0, 2048)
    assert a != cache_key("m", MSGS, 0.7, 0, 2048)
    assert a != cache_key("m", MSGS, 0.0, 1, 2048)
    assert a == cache_key("m", list(MSGS), 0.0, 0, 2048)


def test_request_carries_seed_temperature_and_max_tokens(tmp_path):
    m = model(tmp_path, [ok()], temperature=0.7, seed=3, max_tokens=99)
    m.chat(MSGS)
    body = m.http.bodies[0]
    assert (body["temperature"], body["seed"], body["max_tokens"]) == (0.7, 3, 99)


def test_silent_server_truncation_is_detected(tmp_path):
    # Ollama answered a 29,774-token prompt with prompt_tokens=8194 (HTTP 200).
    m = model(tmp_path, [ok(prompt_tokens=8194)], num_ctx=16384)
    with pytest.raises(ContextLimitError):
        m.chat(MSGS, prefix_tokens=12000)


def test_char_floor_catches_truncation_without_prefix(tmp_path):
    big = [{"role": "user", "content": "x" * 6 * 10000}]  # >= 10,000 tokens by the char floor
    m = model(tmp_path, [ok(prompt_tokens=8194)], num_ctx=16384)
    with pytest.raises(ContextLimitError):
        m.chat(big)


def test_obvious_overflow_is_rejected_before_sending(tmp_path):
    m = model(tmp_path, [], num_ctx=16384)
    with pytest.raises(ContextLimitError):
        m.chat(MSGS, prefix_tokens=20000)
    assert m.http.bodies == []


def test_normal_growth_passes(tmp_path):
    m = model(tmp_path, [ok(prompt_tokens=6100)], num_ctx=16384)
    assert m.chat(MSGS, prefix_tokens=6000).tokens_in == 6100


def test_server_context_error_maps_to_context_limit(tmp_path):
    m = model(tmp_path, [FakeResponse(400, "the request exceeds the available context size")])
    with pytest.raises(ContextLimitError):
        m.chat(MSGS)


def test_client_error_is_not_retried(tmp_path):
    m = model(tmp_path, [FakeResponse(404, "model not found")])
    with pytest.raises(LLMError):
        m.chat(MSGS)
    assert len(m.http.bodies) == 1


def test_min_prompt_tokens_is_a_lower_bound():
    assert min_prompt_tokens([{"role": "user", "content": "a" * 60}]) == 10


def test_tracer_writes_jsonl_and_rejects_unknown_step_fields(tmp_path):
    path = str(tmp_path / "logs" / "run.jsonl")
    with Tracer(path, run_id="r1", seed=0) as tr:
        tr.log("task_start", "task1_1")
        tr.step("task1_1", round=1, raw_response="GET x", parsed_action="GET", success=True)
        with pytest.raises(ValueError):
            tr.step("task1_1", bogus=1)
    recs = read_trace(path)
    assert [r["event"] for r in recs] == ["task_start", "step"]
    assert recs[1]["run_id"] == "r1" and recs[1]["round"] == 1


def test_splits_are_deterministic_disjoint_and_stratified():
    ids = ["task%d_%d" % (c, i) for c in range(1, 11) for i in range(1, 31)]
    a = make_splits.make_splits(ids, seed=0)
    assert a == make_splits.make_splits(list(reversed(ids)), seed=0)
    assert a != make_splits.make_splits(ids, seed=1)
    assert not set(a["dev"]) & set(a["test"])
    assert len(a["dev"]) == 20 and len(a["test"]) == 50
    for cat in range(1, 11):
        assert sum(t.startswith("task%d_" % cat) for t in a["test"]) == 5


def test_committed_splits_match_the_generator():
    from medguardian import mab
    fresh = make_splits.make_splits([c["id"] for c in mab.load_cases()], seed=0)
    for name in ("dev", "test"):
        with open(os.path.join(ROOT, "data", "splits", name + ".json")) as fh:
            assert json.load(fh)["ids"] == fresh[name]
