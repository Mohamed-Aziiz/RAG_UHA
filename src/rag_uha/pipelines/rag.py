"""RAG classique : recherche vectorielle top-k dans ChromaDB, puis génération.

pipe = RAGPipeline(settings)
ans = pipe.answer("Quelles formations propose la FST ?")
"""

from __future__ import annotations

import time

from rag_uha.config import Settings
from rag_uha.embeddings import get_embedder
from rag_uha.indexing.vector_index import open_vector_index
from rag_uha.llm import LLMClient, get_llm
from rag_uha.prompts import ANSWER_SYSTEM, answer_prompt, format_context
from rag_uha.types import Answer, Chunk, RetrievedChunk


class RAGPipeline:
    name = "rag"

    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.llm = llm or get_llm("query", settings)
        self.embedder = get_embedder(settings)
        self.index = open_vector_index(settings, self.embedder)
        self.top_k = settings.rag.top_k

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        retriever = self.index.as_retriever(similarity_top_k=top_k or self.top_k)
        out: list[RetrievedChunk] = []
        for rank, nws in enumerate(retriever.retrieve(question), start=1):
            node = nws.node
            md = node.metadata or {}
            chunk = Chunk(
                id=node.node_id,
                doc_id=md.get("doc_id", node.node_id.split("::")[0]),
                text=node.get_content(),
                lang=md.get("lang", "fr"),
                position=int(md.get("position", 0)),
            )
            out.append(RetrievedChunk(chunk=chunk, rank=rank, score=nws.score))
        return out

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        contexts = self.retrieve(question)
        t_ret = time.perf_counter() - t0

        self.llm.reset_usage()
        text = self.llm.complete(
            answer_prompt(question, format_context(contexts)), system=ANSWER_SYSTEM
        )
        usage = self.llm.reset_usage()

        sources = list(dict.fromkeys(rc.chunk.doc_id for rc in contexts))
        return Answer(
            question=question,
            text=text,
            pipeline="rag",
            contexts=contexts,
            sources=sources,
            usage=usage,
            latency_s=time.perf_counter() - t0,
            retrieval_latency_s=t_ret,
            extra={"top_k": self.top_k, "model": self.llm.model},
        )
