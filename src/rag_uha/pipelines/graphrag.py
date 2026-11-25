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
import atexit
import time
import unicodedata
from typing import Any

from rag_uha.chunking import load_chunks
from rag_uha.config import Settings
from rag_uha.embeddings import get_embedder
from rag_uha.indexing.graph_index import init_lightrag, make_lightrag
from rag_uha.llm import LLMClient, get_llm
from rag_uha.prompts import (
    ANSWER_SYSTEM,
    PROMPT_VERSION,
    answer_prompt,
    clean_answer,
    format_context,
)
from rag_uha.types import Answer, Chunk, RetrievedChunk

_LOOP: asyncio.AbstractEventLoop | None = None


def _norme(t: str) -> str:
    """Minuscules, sans accents, espaces simplifiés (pour compter un mot-clé dans les passages)."""
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def _run(coro):
    """Exécute une coroutine dans UNE boucle asyncio persistante.

    `asyncio.run` créait une boucle neuve à chaque question : LightRAG relançait alors ses
    workers d'embedding (« bound to a different event loop ») et le client HTTP asynchrone,
    ce qui ajoutait de la latence au GraphRAG et faussait la mesure des performances.
    Depuis un notebook (boucle déjà active), on passe par nest_asyncio.
    """
    global _LOOP
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if _LOOP is None or _LOOP.is_closed():
            _LOOP = asyncio.new_event_loop()
        return _LOOP.run_until_complete(coro)
    import nest_asyncio

    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


def _shutdown_loop() -> None:
    """A la fin du programme : annule les taches de fond de LightRAG (workers, controle de sante)
    puis ferme la boucle. Sans cela, Python affiche « Task was destroyed but it is pending »."""
    loop = _LOOP
    if loop is None or loop.is_closed():
        return
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    for t in pending:
        t.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


atexit.register(_shutdown_loop)


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
        # Textes normalisés du corpus : sert à mesurer la part des passages contenant un mot-clé.
        self._textes = [_norme(c.text) for c in self.chunks_by_id.values()]
        self._parts: dict[str, float] = {}
        self.last_keywords: dict[str, list[str]] = {}

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

    # ---- mots-clés génériques (correctif d'Amar, 25/11) -------------------------------------
    # LightRAG cherche les ENTITÉS avec les mots-clés « bas niveau ». Un mot présent presque
    # partout (« UHA » : 94 % des passages) ramène l'entité la plus connectée du graphe et noie
    # le reste. On mesure la part des passages qui contiennent chaque mot-clé ; au-delà de
    # `generic_keyword_share`, un mot-clé bas niveau passe en haut niveau (il reste dans la
    # recherche, mais ne sert plus à chercher les entités). Rien n'est retiré de la question, et
    # la règle est mesurée sur le corpus, pas écrite pour un mot. C'est l'équivalent de l'IDF de
    # BM25 côté RAG. `generic_keyword_share: 0` désactive le correctif.
    def _part_des_passages(self, mot: str) -> float:
        k = _norme(mot)
        if k not in self._parts:
            n = sum(1 for t in self._textes if k in t)
            self._parts[k] = n / len(self._textes) if self._textes else 0.0
        return self._parts[k]

    def _reequilibrer(self, hl: list[str], ll: list[str]) -> tuple[list[str], list[str]]:
        seuil = self.settings.graphrag.generic_keyword_share
        if seuil <= 0 or not self._textes:
            return hl, ll
        generiques = [k for k in ll if self._part_des_passages(k) >= seuil]
        ll2 = [k for k in ll if k not in generiques]
        hl2 = list(dict.fromkeys(hl + generiques))
        if not ll2:  # plus rien pour chercher les entités : on prend les mots-clés précis
            ll2 = [k for k in hl if self._part_des_passages(k) < seuil]
        return hl2, ll2

    def _mots_cles(self, question: str, param) -> tuple[list[str], list[str]]:
        """Mots-clés extraits par LightRAG lui-même (même prompt, même cache que aquery_data)."""
        from lightrag.operate import extract_keywords_only

        return _run(
            extract_keywords_only(
                question, param, self.rag._build_global_config(), self.rag.llm_response_cache
            )
        )

    def retrieve_data(self, question: str, mode: str | None = None) -> dict[str, Any]:
        param = self._param(mode)
        if self.settings.graphrag.generic_keyword_share > 0 and self._textes:
            hl, ll = self._mots_cles(question, param)
            hl2, ll2 = self._reequilibrer(list(hl), list(ll))
            self.last_keywords = {"hl_brut": list(hl), "ll_brut": list(ll), "hl": hl2, "ll": ll2}
            param.hl_keywords, param.ll_keywords = hl2, ll2
        res = _run(self.rag.aquery_data(question, param=param))
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

        graph_context = self._graph_block(data)
        context_block = graph_context
        if contexts:
            context_block += "\n\nSource passages:\n" + format_context(contexts)
        text = clean_answer(
            self.llm.complete(answer_prompt(question, context_block), system=ANSWER_SYSTEM)
        )
        usage = kw_usage + self.llm.reset_usage()

        sources = list(dict.fromkeys(rc.chunk.doc_id for rc in contexts))
        # Repli : seulement si AUCUN passage n'a ete remonte, on cite les documents
        # attaches aux entites / relations. Inconditionnel, ce repli listait presque
        # tout le corpus (96 a 145 documents sur 162), les entites generiques etant
        # reliees a quasiment tous les documents.
        if not contexts:
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
                "prompt": PROMPT_VERSION,
                "entities": [
                    {"name": e["entity_name"], "type": e.get("entity_type")}
                    for e in data.get("entities", [])
                ],
                "relations": [
                    {"src": r["src_id"], "tgt": r["tgt_id"]} for r in data.get("relationships", [])
                ],
                "keywords": res.get("metadata", {}).get("keywords", {}),
                "keywords_reequilibrage": dict(self.last_keywords),
                # Texte des entites / relations reellement envoye au LLM : le juge doit le voir
                # aussi, sinon une affirmation tiree du graphe compte comme une hallucination.
                "graph_context": graph_context,
            },
        )
