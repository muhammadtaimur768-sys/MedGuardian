"""Local LLM access (Ollama's OpenAI-compatible endpoint) with a disk response cache.

* No API key is needed or read: every Phase 2 model runs locally under Ollama.
* Responses are cached on disk keyed by (model, messages, temperature, seed, max_tokens), so a
  crashed or interrupted run resumes instantly and re-runs never repeat finished work (plan D7).
* A prompt that does not fit the model's context window raises ContextLimitError, which the
  runner records as AGENT_CONTEXT_LIMIT exactly as upstream does.

Why overflow detection is needed: Ollama does NOT reject an over-long prompt. Measured on
2026-11-09 (Ollama 0.40.15, num_ctx 16384): a 29,774-token prompt returned HTTP 200 after being
cut to 8,194 tokens (first 4 tokens + the tail), and the model answered the mangled prompt.
So overflow is detected two ways:
  1. before sending, when a safe lower bound on the prompt size already exceeds num_ctx;
  2. after the reply, when the server reports fewer prompt tokens than the conversation must
     contain (it only grows, so it cannot be shorter than the previous prompt plus reply, nor
     shorter than its character length / CHARS_PER_TOKEN_CEILING).
"""
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    ".cache", "llm")


# No tokenizer we use packs more than 6 characters into one token on English or FHIR JSON
# (typical: 3-4), so chars / 6 is a safe lower bound on a prompt's token count.
CHARS_PER_TOKEN_CEILING = 6
# Re-tokenising earlier turns inside a chat template can shift counts by a few tokens.
PREFIX_SLACK_TOKENS = 64


def min_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    return sum(len(m.get("content") or "") for m in messages) // CHARS_PER_TOKEN_CEILING


class LLMError(RuntimeError):
    """The model server failed after retries."""


class ContextLimitError(LLMError):
    """The conversation no longer fits the model's context window."""


@dataclass
class ChatResult:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool = False
    finish_reason: Optional[str] = None
    raw_usage: Dict[str, Any] = field(default_factory=dict)


def cache_key(model: str, messages: List[Dict[str, str]], temperature: float, seed: Optional[int],
              max_tokens: int) -> str:
    blob = json.dumps({"model": model, "messages": messages, "temperature": temperature,
                       "seed": seed, "max_tokens": max_tokens}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _looks_like_context_error(status: int, text: str) -> bool:
    t = text.lower()
    return status in (400, 413, 500) and any(
        k in t for k in ("context", "n_ctx", "too long", "exceeds", "token limit"))


class ChatModel:
    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL, num_ctx: int = 16384,
                 temperature: float = 0.0, max_tokens: int = 2048, seed: Optional[int] = 0,
                 cache_dir: Optional[str] = DEFAULT_CACHE_DIR, timeout: float = 1200,
                 max_retries: int = 4, session: Optional[requests.Session] = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = session or requests.Session()

    # -- cache -------------------------------------------------------------------------------
    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, key[:2], key + ".json")

    def _cache_get(self, key: str) -> Optional[ChatResult]:
        path = self._cache_path(key)
        if not path or not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["cached"] = True
        return ChatResult(**data)

    def _cache_put(self, key: str, res: ChatResult) -> None:
        path = self._cache_path(key)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(res), fh, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache entry

    # -- calls -------------------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], prefix_tokens: int = 0) -> ChatResult:
        """Send one chat turn.

        `prefix_tokens` is the previous round's prompt + completion tokens for the same
        conversation; the new prompt contains all of it, so it cannot legitimately be shorter.
        """
        floor = max(prefix_tokens - PREFIX_SLACK_TOKENS, min_prompt_tokens(messages))
        if self.num_ctx and floor >= self.num_ctx:
            raise ContextLimitError("prompt needs >= %d tokens; num_ctx=%d" % (floor, self.num_ctx))
        key = cache_key(self.model, messages, self.temperature, self.seed, self.max_tokens)
        res = self._cache_get(key)
        if res is None:
            res = self._request(messages)
            self._cache_put(key, res)
        if self.num_ctx and (res.tokens_in >= self.num_ctx or res.tokens_in < floor):
            raise ContextLimitError("server truncated the prompt: prompt_tokens=%d, expected >= %d "
                                    "(num_ctx=%d)" % (res.tokens_in, floor, self.num_ctx))
        return res

    def _request(self, messages: List[Dict[str, str]]) -> ChatResult:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "stream": False}
        if self.seed is not None:
            body["seed"] = self.seed
        delay = 5.0
        last_err = ""
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                r = self.http.post(self.base_url + "/chat/completions", json=body, timeout=self.timeout)
            except requests.RequestException as e:  # server restarting, sleep/wake, etc.
                last_err = "%s: %s" % (type(e).__name__, e)
                time.sleep(delay)
                delay *= 2
                continue
            latency_ms = (time.time() - t0) * 1000
            if r.status_code != 200:
                if _looks_like_context_error(r.status_code, r.text):
                    raise ContextLimitError(r.text[:300])
                last_err = "HTTP %d: %s" % (r.status_code, r.text[:300])
                if r.status_code >= 500 and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(last_err)
            data = r.json()
            usage = data.get("usage") or {}
            choice = data["choices"][0]
            return ChatResult(content=choice["message"].get("content") or "",
                              tokens_in=int(usage.get("prompt_tokens") or 0),
                              tokens_out=int(usage.get("completion_tokens") or 0),
                              latency_ms=round(latency_ms, 1), finish_reason=choice.get("finish_reason"),
                              raw_usage=usage)
        raise LLMError("LLM request failed after %d attempts: %s" % (self.max_retries, last_err))
