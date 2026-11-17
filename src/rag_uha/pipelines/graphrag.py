"""GraphRAG : recherche dans le graphe LightRAG (mode hybrid par défaut), puis génération.

Déroulé d'une question :
1. `aquery_data` : LightRAG extrait les mots-clés (1 appel LLM), cherche les entités et relations
   proches, remonte les chunks liés, et renvoie le tout sans générer de réponse.
2. On construit NOTRE contexte : entités + relations (tableau court) + chunks, et on appelle le
   même prompt de réponse que le RAG avec le même LLM `query`. Ainsi la génération est identique
   des deux côtés ; seule la recherche diffère.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rag_uha.chunking import load_chunks
from rag_uha.config import Settings
from rag_uha.embeddings import get_embedder
from rag_uha.indexing.graph_index import init_lightrag, make_lightrag
from rag_uha.llm import LLMClient, get_llm
from rag_uha.prompts import ANSWER_SYSTEM, answer_prompt, format_context
from rag_uha.types import Answer, Chunk, RetrievedChunk


def _run(coro):
    """Exécute une coroutine, y compris depuis un notebook ou Streamlit (boucle déjà active)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import nest_asyncio

    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


class GraphRAGPipeline:
    name = "graphrag"

    def __init__(
        self, settings: Settings, mode: str | None = None, llm: LLMClient | None = None
    ) -> None:
        self.settings = settings
        self.mode = mode or settings.graphrag.mode
        self.llm = llm or get_llm("query", settings)  # génération ET mots-clés : même rôle `query`
        self.embedder = get_embedder(settings)
        self.rag = make_lightrag(settings, llm=self.llm, embedder=self.embedder)
        _run(init_lightrag(self.rag))
        self.chunks_by_id: dict[str, Chunk] = {}
        if settings.paths.chunks_file.exists():
            self.chunks_by_id = {c.id: c for c in load_chunks(settings.paths.chunks_file)}

    def _param(self, mode: str | None = None):
        from lightrag import QueryParam

        g = self.settings.graphrag
        return QueryParam(
            mode=mode or self.mode,
            top_k=g.top_k,
            max_total_tokens=g.max_total_tokens,
            enable_rerank=False,
        )

    def _to_chunk(self, item: dict[str, Any], rank: int) -> RetrievedChunk:
        cid = item.get("file_path") or item.get("chunk_id") or f"unknown::{rank}"
        known = self.chunks_by_id.get(cid)
        if known is None:
            doc_id = cid.split("::")[0]
            known = Chunk(
                id=cid, doc_id=doc_id, text=item.get("content", ""), lang="fr", position=0
            )
        return RetrievedChunk(chunk=known, rank=rank)

    def retrieve_data(self, question: str, mode: str | None = None) -> dict[str, Any]:
        res = _run(self.rag.aquery_data(question, param=self._param(mode)))
        if res.get("status") != "success":
            raise RuntimeError(f"LightRAG : {res.get('message')}")
        return res

    def retrieve(self, question: str, mode: str | None = None) -> list[RetrievedChunk]:
        data = self.retrieve_data(question, mode)["data"]
        return [self._to_chunk(c, i) for i, c in enumerate(data.get("chunks", []), start=1)]

    @staticmethod
    def _graph_block(data: dict[str, Any], max_entities: int = 20, max_relations: int = 20) -> str:
        ents = data.get("entities", [])[:max_entities]
        rels = data.get("relationships", [])[:max_relations]
        lines = []
        if ents:
            lines.append("Knowledge graph — entities:")
            lines += [
                f"- {e['entity_name']} ({e.get('entity_type', '?')}): {e.get('description', '')[:300]}"
                for e in ents
            ]
        if rels:
            lines.append("\nKnowledge graph — relations:")
            lines += [
                f"- {r['src_id']} -> {r['tgt_id']}: {r.get('description', '')[:300]}" for r in rels
            ]
        return "\n".join(lines)

    def answer(self, question: str, mode: str | None = None) -> Answer:
        t0 = time.perf_counter()
        self.llm.reset_usage()
        res = self.retrieve_data(question, mode)
        kw_usage = self.llm.reset_usage()  # appel d'extraction des mots-clés
        data = res["data"]
        contexts = [self._to_chunk(c, i) for i, c in enumerate(data.get("chunks", []), start=1)]
        t_ret = time.perf_counter() - t0

        context_block = self._graph_block(data)
        if contexts:
            context_block += "\n\nSource passages:\n" + format_context(contexts)
        text = self.llm.complete(answer_prompt(question, context_block), system=ANSWER_SYSTEM)
        usage = kw_usage + self.llm.reset_usage()

        sources = list(dict.fromkeys(rc.chunk.doc_id for rc in contexts))
        # documents cités via les entités / relations mais sans chunk remonté
        for item in data.get("entities", []) + data.get("relationships", []):
            for fp in str(item.get("file_path", "")).split("<SEP>"):
                d = fp.split("::")[0]
                if d and d != "unknown_source" and d not in sources:
                    sources.append(d)

        return Answer(
            question=question,
            text=text,
            pipeline="graphrag",
            contexts=contexts,
            sources=sources,
            usage=usage,
            latency_s=time.perf_counter() - t0,
            retrieval_latency_s=t_ret,
            extra={
                "mode": mode or self.mode,
                "model": self.llm.model,
                "entities": [
                    {"name": e["entity_name"], "type": e.get("entity_type")}
                    for e in data.get("entities", [])
                ],
                "relations": [
                    {"src": r["src_id"], "tgt": r["tgt_id"]} for r in data.get("relationships", [])
                ],
                "keywords": res.get("metadata", {}).get("keywords", {}),
            },
        )
