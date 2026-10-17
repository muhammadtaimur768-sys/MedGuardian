"""Provider-agnostic chat client with tool calling.

Uses the OpenAI-compatible /chat/completions API via raw requests, so it works with Gemini's
OpenAI endpoint, Groq, OpenRouter, Ollama and OpenAI. Configure with env vars (see .env.example).
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


class LLMResponse:
    def __init__(self, message: Dict[str, Any], usage: Dict[str, Any], latency_ms: float):
        self.message = message
        self.usage = usage or {}
        self.latency_ms = latency_ms

    @property
    def content(self) -> Optional[str]:
        return self.message.get("content")

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        out = []
        for tc in self.message.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {"_unparsed": raw}
            out.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})
        return out


class OpenAICompatClient:
    def __init__(self, base_url: str = None, model: str = None, api_key: str = None,
                 timeout: float = 60, max_retries: int = 4, extra_body: Dict[str, Any] = None):
        load_dotenv_if_present()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
        self.api_key = api_key or (os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY")
                                   or os.environ.get("OPENAI_API_KEY"))
        if not self.api_key:
            raise RuntimeError("No API key: set GEMINI_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY) in .env")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body if extra_body is not None else self._default_extra()

    def _default_extra(self) -> Dict[str, Any]:
        # Gemini thinking tokens count against max_tokens; keep reasoning low for tool loops.
        return {"reasoning_effort": "low"} if "generativelanguage" in self.base_url else {}

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None,
             temperature: float = 0.0) -> LLMResponse:
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        body.update(self.extra_body)
        if tools:
            body["tools"] = tools
        headers = {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}
        delay = 4.0
        for attempt in range(self.max_retries):
            t0 = time.time()
            r = requests.post(self.base_url + "/chat/completions", json=body, headers=headers, timeout=self.timeout)
            if r.status_code in (429, 500, 503) and attempt < self.max_retries - 1:
                time.sleep(delay)  # rate limit / transient error: back off and retry
                delay *= 2
                continue
            r.raise_for_status()
            data = r.json()
            return LLMResponse(data["choices"][0]["message"], data.get("usage"), (time.time() - t0) * 1000)
        raise RuntimeError("LLM request failed after %d attempts" % self.max_retries)


def make_client(**kw) -> OpenAICompatClient:
    load_dotenv_if_present()
    provider = os.environ.get("LLM_PROVIDER", "openai_compat")
    if provider != "openai_compat":
        raise NotImplementedError("Only openai_compat is implemented in Phase 1 (got %s)" % provider)
    return OpenAICompatClient(**kw)
