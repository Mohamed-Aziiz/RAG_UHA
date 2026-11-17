"""Doublures communes : embedder déterministe (sans réseau) et LLM factice."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from rag_uha.config import EmbeddingsSettings, LLMSettings, RetrySettings
from rag_uha.embeddings import Embedder
from rag_uha.llm import LLMClient
from rag_uha.types import Usage


class HashEmbedder(Embedder):
    """Vecteur = sac de mots haché : deux textes partageant des mots sont proches. Suffisant pour tester."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(EmbeddingsSettings(dim=dim, batch_size=16))
        self.model = "hash-test"

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return out


class FakeLLM(LLMClient):
    """Renvoie une réponse canonique sans réseau ; compte des tokens approximatifs."""

    def __init__(self, reply: str = "Réponse factice [doc]") -> None:
        cfg = LLMSettings(base_url="http://fake", model="fake", api_key_env=None)
        self.role, self.cfg, self.retry, self.model = "query", cfg, RetrySettings(), "fake"
        self.usage, self.last_usage, self.reply, self.prompts = Usage(), Usage(), reply, []

    def complete(self, prompt, system=None, history=None, max_tokens=None, **kw):
        self.prompts.append((system, prompt))
        self.last_usage = Usage(len(prompt) // 4, len(self.reply) // 4, 1)
        self.usage = self.usage + self.last_usage
        return self.reply

    async def acomplete(self, prompt, system=None, history=None, max_tokens=None, **kw):
        return self.complete(prompt, system, history, max_tokens, **kw)


@pytest.fixture
def hash_embedder():
    return HashEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()
