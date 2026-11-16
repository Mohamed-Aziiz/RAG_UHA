"""Tests de la couche LLM, sans réseau : on remplace le client OpenAI par un faux.

uv run pytest -q
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from rag_uha.config import LLMSettings, RetrySettings
from rag_uha.llm import LLMClient, _wait_seconds, as_lightrag_llm_func

CFG = LLMSettings(
    provider="groq",
    base_url="https://api.groq.com/openai/v1",
    model="test-model",
    api_key_env=None,
    max_tokens=50,
    extra={"reasoning_effort": "low"},
)
RETRY = RetrySettings(max_attempts=3, min_wait_s=0.01, max_wait_s=0.05)


def _response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://x"))
    return openai.RateLimitError("rate limited", response=resp, body=None)


class FakeCompletions:
    """Renvoie les éléments de `script` dans l'ordre : une exception est levée, un objet renvoyé."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def acreate(self, **kwargs):
        return self.create(**kwargs)


def _client_with(script: list) -> tuple[LLMClient, FakeCompletions]:
    llm = LLMClient("query", CFG, RETRY)
    fake = FakeCompletions(script)
    llm._sync = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake.create))
    )
    llm._async = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake.acreate))
    )
    return llm, fake


def test_complete_counts_usage_and_passes_params():
    llm, fake = _client_with([_response("Bonjour", 12, 3)])
    out = llm.complete("Salut", system="Sois bref")
    assert out == "Bonjour"
    assert llm.usage.prompt_tokens == 12 and llm.usage.completion_tokens == 3
    assert llm.usage.n_calls == 1
    call = fake.calls[0]
    assert call["model"] == "test-model"
    assert call["max_tokens"] == 50
    assert call["extra_body"] == {"reasoning_effort": "low"}
    assert call["messages"][0] == {"role": "system", "content": "Sois bref"}
    assert call["messages"][-1] == {"role": "user", "content": "Salut"}


def test_retry_on_429_then_success():
    llm, fake = _client_with([_rate_limit_error("0.01"), _rate_limit_error(), _response("ok")])
    assert llm.complete("x") == "ok"
    assert len(fake.calls) == 3
    assert llm.usage.n_calls == 1  # seuls les appels réussis comptent


def test_gives_up_after_max_attempts():
    llm, _ = _client_with([_rate_limit_error()] * 3)
    with pytest.raises(openai.RateLimitError):
        llm.complete("x")


def test_no_retry_on_auth_error():
    resp = httpx.Response(401, request=httpx.Request("POST", "https://x"))
    err = openai.AuthenticationError("bad key", response=resp, body=None)
    llm, fake = _client_with([err, _response("jamais")])
    with pytest.raises(openai.AuthenticationError):
        llm.complete("x")
    assert len(fake.calls) == 1


def test_wait_respects_retry_after_header():
    retry = RetrySettings(max_attempts=3, min_wait_s=1, max_wait_s=100)
    assert _wait_seconds(_rate_limit_error("7.66s"), 1, retry) == pytest.approx(8.66)
    assert _wait_seconds(_rate_limit_error("1m2s"), 1, retry) == pytest.approx(63.0)
    assert _wait_seconds(_rate_limit_error("500"), 1, retry) == 100  # borné par max_wait_s


def test_reset_usage():
    llm, _ = _client_with([_response("a", 5, 5), _response("b", 7, 1)])
    llm.complete("1")
    llm.complete("2")
    total = llm.reset_usage()
    assert total.total_tokens == 18 and total.n_calls == 2
    assert llm.usage.n_calls == 0


def test_lightrag_adapter_ignores_unknown_kwargs():
    llm, fake = _client_with([_response("entités")])
    func = as_lightrag_llm_func(llm)
    out = asyncio.run(
        func(
            "texte",
            system_prompt="extrait",
            history_messages=[{"role": "user", "content": "h"}],
            keyword_extraction=False,
            enable_cot=True,
            hashing_kv=None,
            stream=False,
        )
    )
    assert out == "entités"
    msgs = fake.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "user"]
