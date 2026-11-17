"""Tests des embeddings sans réseau : transport httpx simulé."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from rag_uha.config import EmbeddingsSettings, RetrySettings
from rag_uha.embeddings import CloudflareEmbedder

RETRY = RetrySettings(max_attempts=3, min_wait_s=0.01, max_wait_s=0.02)


def _embedder(monkeypatch, handler, batch_size: int = 2) -> tuple[CloudflareEmbedder, list]:
    monkeypatch.setenv("CF_ACCOUNT_ID", "acc")
    monkeypatch.setenv("CF_API_TOKEN", "tok")
    cfg = EmbeddingsSettings(batch_size=batch_size, dim=4)
    emb = CloudflareEmbedder(cfg, RETRY)
    calls: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    emb._client = httpx.Client(transport=httpx.MockTransport(_handler))
    return emb, calls


def _ok(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["text"]
    data = [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]  # vecteurs factices
    return httpx.Response(
        200, json={"success": True, "result": {"shape": [len(texts), 4], "data": data}}
    )


def test_batches_and_normalisation(monkeypatch):
    emb, calls = _embedder(monkeypatch, _ok, batch_size=2)
    v = emb.embed_documents(["a", "bb", "ccc"])
    assert v.shape == (3, 4)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert len(calls) == 2  # 2 + 1
    assert emb.n_texts == 3


def test_retry_on_429(monkeypatch):
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, json={"success": False, "errors": ["quota"]})
        return _ok(request)

    emb, calls = _embedder(monkeypatch, flaky)
    assert emb.embed_query("x").shape == (4,)
    assert len(calls) == 2


def test_400_is_not_retried_and_isolates_bad_text(monkeypatch):
    def handler(request):
        texts = json.loads(request.content)["text"]
        assert json.loads(request.content)["truncate_inputs"] is True
        if any("BAD" in t for t in texts):
            return httpx.Response(
                400, json={"success": False, "errors": [{"code": 3016, "message": "too long"}]}
            )
        return _ok(request)

    emb, calls = _embedder(monkeypatch, handler, batch_size=4)
    with pytest.raises(RuntimeError, match="Cloudflare refuse ce texte"):
        emb.embed_documents(["ok1", "BAD text", "ok2"])
    # 1 appel pour le lot, puis texte par texte jusqu'au fautif : jamais de boucle de retry
    assert len(calls) <= 4


def test_fails_on_bad_shape(monkeypatch):
    def bad(request):
        return httpx.Response(200, json={"success": True, "result": {"data": [[1.0, 0.0]]}})

    emb, _ = _embedder(monkeypatch, bad)
    with pytest.raises(RuntimeError):
        emb.embed_query("x")


def test_empty_input(monkeypatch):
    emb, calls = _embedder(monkeypatch, _ok)
    assert emb.embed_documents([]).shape == (0, 4)
    assert calls == []
