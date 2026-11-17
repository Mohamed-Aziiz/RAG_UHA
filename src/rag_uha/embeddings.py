"""Embeddings BGE-M3 : un seul modèle pour le RAG, le GraphRAG et le juge.

Par défaut, les vecteurs sont calculés par l'API Cloudflare Workers AI (`@cf/baai/bge-m3`) :
rien à télécharger, un collègue peut tester avec une simple clé. Le plan B (`provider: local`)
charge le même modèle avec sentence-transformers sur le GPU — mêmes vecteurs, index compatibles.

Pourquoi BGE-M3 : multilingue (une question en français retrouve un chunk en anglais), 1 024
dimensions, 8 192 tokens de contexte, gratuit sur Cloudflare (10 000 neurons/jour ≈ 9 M tokens).

Usage :
    emb = get_embedder(settings)
    vecs = emb.embed_documents(["texte 1", "texte 2"])   # ndarray (2, 1024), normalisés
    q = emb.embed_query("question")                       # ndarray (1024,)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import numpy as np

from rag_uha.config import EmbeddingsSettings, RetrySettings, Settings, load_settings
from rag_uha.tokens import get_encoding

log = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class CloudflareRequestError(RuntimeError):
    """Erreur 4xx définitive (requête invalide, texte trop long…) : ne pas réessayer."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status} : {body}")
        self.status = status


class Embedder:
    """Interface commune. Les sous-classes implémentent `_embed_batch`."""

    dim: int = 1024
    model: str = ""

    def __init__(self, cfg: EmbeddingsSettings) -> None:
        self.cfg = cfg
        self.dim = cfg.dim
        self.model = cfg.model
        self.n_texts = 0  # statistiques pour le rapport (coût d'indexation)
        self.n_tokens_est = 0  # estimation tiktoken, Cloudflare ne renvoie pas le compte exact
        self._enc = get_encoding("cl100k_base")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:  # (n, dim)
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Vecteurs normalisés (n, dim), calculés par lots de `batch_size`."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.cfg.batch_size]]
            vecs = np.asarray(self._embed_batch(batch), dtype=np.float32)
            if vecs.shape != (len(batch), self.dim):
                raise RuntimeError(
                    f"Forme inattendue {vecs.shape}, attendu ({len(batch)}, {self.dim})"
                )
            out.append(vecs)
            self.n_texts += len(batch)
            self.n_tokens_est += sum(len(self._enc.encode(t)) for t in batch)
        all_vecs = np.vstack(out)
        norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
        return all_vecs / np.clip(norms, 1e-12, None)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))


class CloudflareEmbedder(Embedder):
    """BGE-M3 via l'API REST Workers AI. Lots de ≤ 100 textes, réessais sur 429 / 5xx."""

    def __init__(self, cfg: EmbeddingsSettings, retry: RetrySettings) -> None:
        super().__init__(cfg)
        self.retry = retry
        self.url = (
            f"https://api.cloudflare.com/client/v4/accounts/{cfg.account_id}/ai/run/{cfg.model}"
        )
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {cfg.api_token}"}, timeout=120
        )

    def _post(self, texts: list[str]) -> np.ndarray:
        """Un appel HTTP. 4xx (sauf 408/429) = erreur définitive : on remonte le message de Cloudflare."""
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                r = self._client.post(self.url, json={"text": texts, "truncate_inputs": True})
                if r.status_code in _RETRY_STATUS:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                if r.status_code >= 400:
                    raise CloudflareRequestError(r.status_code, r.text[:500])
                payload = r.json()
                if not payload.get("success", True):
                    raise CloudflareRequestError(r.status_code, str(payload.get("errors"))[:500])
                return np.asarray(payload["result"]["data"], dtype=np.float32)
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                if attempt == self.retry.max_attempts:
                    raise
                wait = min(self.retry.min_wait_s * 2 ** (attempt - 1), self.retry.max_wait_s)
                log.warning(
                    "Cloudflare %s — nouvel essai %d dans %.0fs", type(exc).__name__, attempt, wait
                )
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        try:
            return self._post(texts)
        except CloudflareRequestError as exc:
            if len(texts) == 1:
                raise RuntimeError(
                    f"Cloudflare refuse ce texte ({exc}) : {texts[0][:120]!r}…"
                ) from exc
            # Un seul texte fautif fait échouer tout le lot : on isole le coupable un par un.
            log.warning("Lot de %d textes refusé (%s) : passage texte par texte", len(texts), exc)
            return np.vstack([self._embed_batch([t]) for t in texts])


class LocalEmbedder(Embedder):
    """Plan B : même modèle en local (sentence-transformers, GPU si disponible).

    Nécessite `uv add sentence-transformers torch` (non inclus par défaut : ~2 Go).
    """

    def __init__(self, cfg: EmbeddingsSettings) -> None:
        super().__init__(cfg)
        from sentence_transformers import SentenceTransformer  # import tardif, optionnel

        self._model = SentenceTransformer("BAAI/bge-m3")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, batch_size=self.cfg.batch_size, normalize_embeddings=True)


# ---------------------------------------------------------------------- fabrique
_CACHE: dict[str, Embedder] = {}


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or load_settings()
    cfg = settings.embeddings
    key = f"{cfg.provider}:{cfg.model}"
    if key not in _CACHE:
        if cfg.provider == "cloudflare":
            _CACHE[key] = CloudflareEmbedder(cfg, settings.retry)
        else:
            _CACHE[key] = LocalEmbedder(cfg)
    return _CACHE[key]


# ---------------------------------------------------------------------- adaptateurs
def as_llamaindex_embedding(embedder: Embedder):
    """Objet `BaseEmbedding` pour LlamaIndex (index ChromaDB et recherche du RAG)."""
    from llama_index.core.base.embeddings.base import BaseEmbedding

    class _LlamaIndexEmbedding(BaseEmbedding):
        def __init__(self) -> None:
            super().__init__(model_name=embedder.model, embed_batch_size=embedder.cfg.batch_size)

        @classmethod
        def class_name(cls) -> str:
            return "RagUhaEmbedding"

        def _get_query_embedding(self, query: str) -> list[float]:
            return embedder.embed_query(query).tolist()

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return embedder.embed_query(text).tolist()

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return embedder.embed_documents(texts).tolist()

    return _LlamaIndexEmbedding()


def as_lightrag_embedding_func(embedder: Embedder):
    """`EmbeddingFunc` pour LightRAG (entités, relations, chunks du graphe)."""
    from lightrag.utils import EmbeddingFunc

    async def _embed(texts: list[str], **kwargs: Any) -> np.ndarray:
        return embedder.embed_documents(list(texts))

    return EmbeddingFunc(
        embedding_dim=embedder.dim,
        max_token_size=8192,
        func=_embed,
        model_name=embedder.model,
    )


def as_ragas_embeddings(embedder: Embedder, legacy: bool = False):
    """Embeddings pour RAGAS (answer_relevancy, answer_correctness).

    - `legacy=False` : objet de l'API moderne (`BaseRagasEmbedding`).
    - `legacy=True` : `LangchainEmbeddingsWrapper` autour d'un `Embeddings` LangChain.
    """
    if legacy:
        from langchain_core.embeddings import Embeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        class _LC(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return embedder.embed_documents(texts).tolist()

            def embed_query(self, text: str) -> list[float]:
                return embedder.embed_query(text).tolist()

        return LangchainEmbeddingsWrapper(_LC())

    from ragas.embeddings.base import BaseRagasEmbedding

    class _Modern(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embedder.embed_query(text).tolist()

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.embed_text(text)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return embedder.embed_documents(texts).tolist()

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return self.embed_texts(texts)

    return _Modern()
