"""Couche LLM unique : le seul endroit du projet qui parle à un LLM.

Un `LLMClient` par rôle (indexing, query, judge). Chaque rôle lit son fournisseur et son modèle
dans la config : changer de fournisseur (Groq ↔ Ollama) = changer `base_url` + `model` dans
le YAML, sans toucher au code. Tous les fournisseurs sont compatibles avec l'API OpenAI.

Ce module :
- compte les tokens de chaque appel (`usage`), pour le 4e axe de la comparaison ;
- réessaie avec attente exponentielle sur les erreurs 429 (quota), 413 (requête trop grosse pour
  le quota par minute) et 5xx, en respectant l'en-tête `retry-after` quand Groq le donne ;
- limite la concurrence des appels asynchrones (`max_concurrency`) ;
- fournit trois adaptateurs : LlamaIndex (RAG), LightRAG (GraphRAG) et RAGAS (juge).

Usage :
    llm = get_llm("query", settings)
    text = llm.complete("Quelle est la capitale de l'Alsace ?", system="Réponds en une phrase.")
    print(llm.usage)           # Usage(prompt_tokens=…, completion_tokens=…, n_calls=1)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Literal

import openai
from openai import AsyncOpenAI, OpenAI

from rag_uha.config import LLMSettings, RetrySettings, Settings, load_settings
from rag_uha.types import Usage

log = logging.getLogger(__name__)

Role = Literal["indexing", "query", "judge"]

# Erreurs sur lesquelles on réessaie : quota (429), requête trop grosse pour le débit (413),
# surcharge serveur (5xx), problèmes réseau.
_RETRY_STATUS = {408, 409, 413, 429, 500, 502, 503, 504}


def _wait_seconds(exc: Exception, attempt: int, retry: RetrySettings) -> float:
    """Durée d'attente avant le prochain essai.

    Groq renvoie souvent `retry-after` (en secondes) sur les 429 : on le respecte.
    Sinon : attente exponentielle bornée, avec un peu de hasard pour ne pas repartir pile au
    même instant.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            raw = headers.get(name)
            if raw:
                try:
                    # formats vus : "7.66s", "1m2s", "12"
                    s = raw.strip().lower().replace("ms", "")
                    total = 0.0
                    if "m" in s:
                        mins, _, rest = s.partition("m")
                        total += float(mins) * 60
                        s = rest
                    total += float(s.rstrip("s") or 0)
                    return min(max(total + 1.0, retry.min_wait_s), retry.max_wait_s)
                except ValueError:
                    pass
    expo = retry.min_wait_s * (2 ** (attempt - 1))
    return min(expo, retry.max_wait_s) * random.uniform(0.8, 1.2)


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRY_STATUS
    return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))


class LLMClient:
    """Client compatible OpenAI (Groq, Ollama, OpenAI) pour un rôle donné."""

    def __init__(self, role: Role, cfg: LLMSettings, retry: RetrySettings) -> None:
        self.role = role
        self.cfg = cfg
        self.retry = retry
        self.model = cfg.model
        self.usage = Usage()
        self.last_usage = Usage()
        self._sync = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, max_retries=0, timeout=120)
        self._async = AsyncOpenAI(
            base_url=cfg.base_url, api_key=cfg.api_key, max_retries=0, timeout=120
        )
        self._sem = asyncio.Semaphore(max(1, cfg.max_concurrency))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _messages(
        prompt: str, system: str | None, history: list[dict[str, str]] | None
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _params(self, max_tokens: int | None, **overrides: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if self.cfg.extra:
            params["extra_body"] = dict(self.cfg.extra)  # ex. reasoning_effort pour gpt-oss
        # Paramètres OpenAI standard passés par les bibliothèques (response_format, stop…)
        for k in ("response_format", "stop", "seed"):
            if k in overrides and overrides[k] is not None:
                params[k] = overrides[k]
        return params

    def _record(self, response: Any) -> str:
        u = getattr(response, "usage", None)
        used = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            n_calls=1,
        )
        self.last_usage = used
        self.usage = self.usage + used
        content = response.choices[0].message.content
        return (content or "").strip()

    # ------------------------------------------------------------------ appels
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> str:
        """Appel synchrone avec réessais. Renvoie le texte de la réponse."""
        msgs = self._messages(prompt, system, history)
        params = self._params(max_tokens, **kw)
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                return self._record(self._sync.chat.completions.create(messages=msgs, **params))
            except Exception as exc:
                if not _should_retry(exc) or attempt == self.retry.max_attempts:
                    raise
                wait = _wait_seconds(exc, attempt, self.retry)
                log.warning(
                    "[%s/%s] %s — nouvel essai %d/%d dans %.0fs",
                    self.role,
                    self.model,
                    type(exc).__name__,
                    attempt,
                    self.retry.max_attempts,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError("unreachable")

    async def acomplete(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> str:
        """Appel asynchrone avec réessais et limite de concurrence (utilisé par LightRAG)."""
        msgs = self._messages(prompt, system, history)
        params = self._params(max_tokens, **kw)
        async with self._sem:
            for attempt in range(1, self.retry.max_attempts + 1):
                try:
                    resp = await self._async.chat.completions.create(messages=msgs, **params)
                    return self._record(resp)
                except Exception as exc:
                    if not _should_retry(exc) or attempt == self.retry.max_attempts:
                        raise
                    wait = _wait_seconds(exc, attempt, self.retry)
                    log.warning(
                        "[%s/%s] %s — nouvel essai %d/%d dans %.0fs",
                        self.role,
                        self.model,
                        type(exc).__name__,
                        attempt,
                        self.retry.max_attempts,
                        wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    def reset_usage(self) -> Usage:
        """Renvoie le cumul puis le remet à zéro (pour mesurer une question ou une indexation)."""
        total, self.usage = self.usage, Usage()
        return total


# ---------------------------------------------------------------------- fabrique
_CACHE: dict[tuple[Role, str, str], LLMClient] = {}


def get_llm(role: Role, settings: Settings | None = None) -> LLMClient:
    """Un client par rôle, réutilisé tant que la config ne change pas."""
    settings = settings or load_settings()
    cfg = settings.llm(role)
    key = (role, cfg.base_url, cfg.model)
    if key not in _CACHE:
        _CACHE[key] = LLMClient(role, cfg, settings.retry)
    return _CACHE[key]


# ---------------------------------------------------------------------- adaptateurs
def as_llamaindex_llm(llm: LLMClient):
    """Objet LLM pour LlamaIndex (pipeline RAG).

    LlamaIndex fait ses propres appels HTTP via `OpenAILike` : on lui passe les mêmes
    paramètres (base_url, modèle, température, extra) et ses propres réessais sur 429.
    Les tokens de ces appels ne passent pas par `llm.usage` : le pipeline RAG mesure
    la génération en appelant `llm.complete` directement (voir pipelines/rag.py).
    Cet adaptateur ne sert donc qu'aux composants LlamaIndex qui exigent un LLM.
    """
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=llm.model,
        api_base=llm.cfg.base_url,
        api_key=llm.cfg.api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=llm.cfg.temperature,
        max_tokens=llm.cfg.max_tokens,
        additional_kwargs={"extra_body": dict(llm.cfg.extra)} if llm.cfg.extra else {},
        max_retries=llm.retry.max_attempts,
        timeout=120,
        context_window=128_000,
    )


def as_lightrag_llm_func(llm: LLMClient):
    """Fonction asynchrone au format attendu par LightRAG (`llm_model_func`).

    LightRAG l'appelle avec `prompt`, puis des mots-clés variables selon la version :
    `system_prompt`, `history_messages`, `keyword_extraction`, `enable_cot`, `stream`,
    `hashing_kv`… On ne garde que ce qui nous concerne, le reste est ignoré.
    """

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        if kwargs.get("stream"):
            # Le streaming n'est utilisé que par le serveur web de LightRAG, pas par nous.
            log.debug("stream=True ignoré : réponse renvoyée d'un bloc")
        return await llm.acomplete(
            prompt,
            system=system_prompt,
            history=history_messages or None,
            max_tokens=kwargs.get("max_tokens"),
            response_format=kwargs.get("response_format"),
        )

    return llm_model_func


def as_ragas_llm(llm: LLMClient, legacy: bool = False):
    """Juge pour RAGAS.

    - `legacy=False` (API RAGAS 0.3+/0.4, `ragas.metrics.collections`) : `llm_factory` avec un
      client OpenAI pointé sur Groq. Pas de LangChain nécessaire.
    - `legacy=True` (ancienne API `ragas.evaluate` + `ragas.metrics.faithfulness`…) :
      `LangchainLLMWrapper(ChatOpenAI(...))`, nécessite `langchain-openai`.
    Le choix définitif se fait à la tâche T13, après le test sur 2 questions.
    """
    if legacy:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        chat = ChatOpenAI(
            model=llm.model,
            base_url=llm.cfg.base_url,
            api_key=llm.cfg.api_key,
            temperature=llm.cfg.temperature,
            max_tokens=llm.cfg.max_tokens,
            max_retries=llm.retry.max_attempts,
            extra_body=dict(llm.cfg.extra) or None,
        )
        return LangchainLLMWrapper(chat)

    from ragas.llms import llm_factory

    return llm_factory(
        model=llm.model,
        provider="openai",
        client=AsyncOpenAI(
            base_url=llm.cfg.base_url, api_key=llm.cfg.api_key, max_retries=llm.retry.max_attempts
        ),
        temperature=llm.cfg.temperature,
        max_tokens=llm.cfg.max_tokens,
    )
