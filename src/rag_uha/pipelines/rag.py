"""RAG classique : recherche vectorielle top-k dans ChromaDB, puis génération.

pipe = RAGPipeline(settings)
ans = pipe.answer("Quelles formations propose la FST ?")
"""

from __future__ import annotations

import logging
import time

from rag_uha.chunking import load_chunks
from rag_uha.config import Settings
from rag_uha.embeddings import get_embedder
from rag_uha.indexing.vector_index import open_vector_index
from rag_uha.lexical import BM25Index, rrf
from rag_uha.llm import LLMClient, get_llm
from rag_uha.prompts import ANSWER_SYSTEM, PROMPT_VERSION, answer_prompt, format_context
from rag_uha.rerank import get_reranker
from rag_uha.types import Answer, Chunk, RetrievedChunk

log = logging.getLogger(__name__)


class RAGPipeline:
    name = "rag"

    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.llm = llm or get_llm("query", settings)
        self.embedder = get_embedder(settings)
        self.index = open_vector_index(settings, self.embedder)
        self.top_k = settings.rag.top_k
        self.reranker = get_reranker(settings) if settings.rag.rerank else None
        self.chunks_by_id: dict[str, Chunk] = {}
        self.bm25: BM25Index | None = None
        if settings.rag.hybrid and settings.paths.chunks_file.exists():
            chunks = load_chunks(settings.paths.chunks_file)
            self.chunks_by_id = {c.id: c for c in chunks}
            self.bm25 = BM25Index(chunks)
            log.info("BM25 : %d chunks indexes", len(chunks))

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        # Avec reranking ou fusion on remonte beaucoup plus de candidats.
        n = max(k, self.settings.rag.rerank_candidates) if (self.reranker or self.bm25) else k
        retriever = self.index.as_retriever(similarity_top_k=n)
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
        if self.bm25 is not None:
            out = self._fuse(question, out)
        if self.reranker and len(out) > 1:
            out = self._rerank(question, out, k)
        return out[:k]

    def _fuse(self, question: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Fusionne le classement vectoriel et le classement BM25.

        - "interleave" (defaut) : 1er vectoriel, puis les `bm25_slots` meilleurs BM25, puis
          la suite du vectoriel, sans doublon. Le meilleur resultat lexical est garanti dans le
          contexte, et le vectoriel garde le reste. Avec une place BM25 sur deux, les questions
          inter-langues tombaient de 0,25 a 0 : BM25 ne trouve rien d'une langue a l'autre, et
          son bruit chassait les resultats vectoriels, les seuls qui marchent entre langues.
        - "rrf" : Reciprocal Rank Fusion. Defaut constate sur notre corpus : un chunk 1er en
          BM25 mais absent des candidats vectoriels (1/61 = 0,016) est battu par des chunks
          moyens dans les deux listes (~0,030). Or c'est precisement le cas des chunks-listes
          que seul le lexical retrouve.
        """
        assert self.bm25 is not None
        vect = [h.chunk.id for h in hits]
        lex = [cid for cid, _ in self.bm25.search(question, self.settings.rag.bm25_candidates)]
        connus = {h.chunk.id: h.chunk for h in hits} | self.chunks_by_id

        if self.settings.rag.fusion == "rrf":
            scores = rrf([vect, lex], k=self.settings.rag.rrf_k)
            ordre = sorted(scores, key=lambda cid: -scores[cid])
        else:
            n_lex = self.settings.rag.bm25_slots
            ordre, vus = [], set()
            for cid in vect[:1] + lex[:n_lex] + vect[1:] + lex[n_lex:]:
                if cid not in vus:
                    vus.add(cid)
                    ordre.append(cid)
        return [
            RetrievedChunk(chunk=connus[cid], rank=rang, score=1.0 / rang)
            for rang, cid in enumerate((c for c in ordre if c in connus), start=1)
        ]

    def _rerank(self, question: str, hits: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Reclasse les candidats par cross-encoder. En cas d'echec : ordre vectoriel conserve."""
        try:
            scores = self.reranker.scores(question, [h.chunk.text for h in hits])
        except Exception as exc:  # noqa: BLE001 — le reranker ne doit jamais casser une reponse
            log.warning("Reranker indisponible (%s) : ordre vectoriel conserve", exc)
            return hits[:k]
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            RetrievedChunk(chunk=hits[i].chunk, rank=rank, score=scores[i])
            for rank, i in enumerate(order, start=1)
        ]

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
            extra={
                "top_k": self.top_k,
                "rerank": self.reranker is not None,
                "hybrid": self.bm25 is not None,
                "model": self.llm.model,
                "prompt": PROMPT_VERSION,
            },
        )
