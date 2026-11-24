"""Reclassement des passages recuperes (cross-encoder BGE via Cloudflare Workers AI).

Pourquoi : la recherche vectorielle compare la question et le passage *separement*, via
deux embeddings. Sur un corpus homogene comme le notre, tous les passages se ressemblent
— les scores BGE-M3 tiennent dans une bande de 0,06 — et le bon passage peut sortir au
rang 16 ou 31. Un cross-encoder lit la question ET le passage ensemble, ce qui lui permet
de reperer une correspondance litterale noyee dans un long texte.

Le pipeline devient : ChromaDB remonte `rerank_candidates` passages (gratuit, pas de LLM),
le reranker les note, on garde les `top_k` meilleurs. Le LLM recoit toujours `top_k`
passages : le cout de generation ne change pas.

Limite a connaitre : bge-reranker-base a une fenetre de 512 tokens. Sur un chunk de
1 200 tokens, il ne voit que le debut. C'est un argument de plus pour des chunks courts.

    rr = get_reranker(settings)
    scores = rr.scores("ma question", ["passage 1", "passage 2"])  # un score par passage
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from rag_uha.config import Settings, load_settings

log = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_MAX_CHARS = 2400  # ~600 tokens : au-dela, le modele tronque de toute facon a 512


class Reranker:
    """Cross-encoder BGE via l'API REST Workers AI."""

    def __init__(self, settings: Settings) -> None:
        cfg = settings.embeddings  # memes identifiants Cloudflare que les embeddings
        self.model = settings.rag.rerank_model
        self.batch_size = settings.rag.rerank_batch
        self.retry = settings.retry
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{cfg.account_id}/ai/run/{self.model}"
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {cfg.api_token}"}, timeout=60
        )
        self.n_calls = 0

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _parse(payload: Any, n: int) -> list[float]:
        """La doc Cloudflare ne fixe pas la forme de la reponse : on accepte les variantes.

        Formes gerees : {"result": {"response": [{"id": i, "score": s}, ...]}},
        une liste de dicts directement, ou une liste de nombres dans l'ordre des contextes.
        """
        node: Any = payload
        for key in ("result", "response"):
            if isinstance(node, dict) and key in node:
                node = node[key]
        if not isinstance(node, list):
            raise ValueError(f"reponse inattendue : {str(payload)[:200]}")

        scores: list[float | None] = [None] * n
        for pos, item in enumerate(node):
            if isinstance(item, dict):
                idx = item.get("id", item.get("index", pos))
                val = item.get("score", item.get("relevance_score", item.get("value")))
                if val is None:
                    continue
                i = int(idx)
                if 0 <= i < n:
                    scores[i] = float(val)
            elif isinstance(item, (int, float)) and pos < n:
                scores[pos] = float(item)
        if all(s is None for s in scores):
            raise ValueError(f"aucun score exploitable : {str(payload)[:200]}")
        return [s if s is not None else float("-inf") for s in scores]

    # ------------------------------------------------------------------ appel
    def _post(self, query: str, texts: list[str]) -> list[float]:
        body = {
            "query": query,
            "contexts": [{"text": t[:_MAX_CHARS]} for t in texts],
            "top_k": len(texts),  # on veut un score pour chaque passage
        }
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                r = self._client.post(self.url, json=body)
                if r.status_code in _RETRY_STATUS:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code} : {r.text[:300]}")
                self.n_calls += 1
                return self._parse(r.json(), len(texts))
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                if attempt == self.retry.max_attempts:
                    raise
                wait = min(self.retry.min_wait_s * 2 ** (attempt - 1), self.retry.max_wait_s)
                log.warning("Reranker %s — nouvel essai %d dans %.0fs", type(exc).__name__, attempt, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def scores(self, query: str, texts: list[str]) -> list[float]:
        """Un score par passage, dans l'ordre d'entree."""
        if not texts:
            return []
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post(query, texts[i : i + self.batch_size]))
        return out


_CACHE: dict[str, Reranker] = {}


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or load_settings()
    key = settings.rag.rerank_model
    if key not in _CACHE:
        _CACHE[key] = Reranker(settings)
    return _CACHE[key]
