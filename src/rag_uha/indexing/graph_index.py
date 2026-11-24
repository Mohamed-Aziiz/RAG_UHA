"""Index GraphRAG : chunks → extraction d'entités et relations par le LLM → graphe (LightRAG).

Choix :
- Chaque chunk commun est inséré comme un document LightRAG à part entière, avec
  `file_paths=[chunk.id]`. Un texte de 1 200 tokens n'est pas redécoupé (taille de chunk LightRAG
  = 1 200), et les contextes récupérés portent `file_path = "<doc_id>::<n>"` : la source est
  connue sans travail supplémentaire. Les deux architectures voient donc exactement les mêmes chunks.
- Stockages par défaut : graphe NetworkX (GraphML), vecteurs NanoVectorDB, KV JSON. Pas de serveur.
- Cache LLM activé : relancer `build_graph` après une coupure (quota, erreur) reprend sans repayer
  les chunks déjà extraits ; LightRAG ignore de plus les documents déjà insérés (même contenu).
- `metadata.json` enregistre modèles, paramètres, tokens et durée : c'est le « coût d'indexation »
  du 4e axe de la comparaison.

    report = build_graph(chunks, settings, limit=3)   # test sur 3 documents
    report = build_graph(chunks, settings)            # indexation complète (reprise automatique)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

from rag_uha.config import Settings
from rag_uha.embeddings import Embedder, as_lightrag_embedding_func, get_embedder
from rag_uha.indexing.vector_index import write_metadata
from rag_uha.llm import LLMClient, as_lightrag_llm_func, get_llm
from rag_uha.prompts import UHA_ENTITY_TYPES_GUIDANCE
from rag_uha.types import Chunk, Usage

log = logging.getLogger(__name__)

GRAPHML_NAME = "graph_chunk_entity_relation.graphml"


@dataclass
class GraphBuildReport:
    n_chunks_requested: int
    n_chunks_total: int
    n_entities: int
    n_relations: int
    usage: Usage
    duration_s: float
    model: str


def make_lightrag(
    settings: Settings, llm: LLMClient | None = None, embedder: Embedder | None = None
):
    """Instance LightRAG configurée (non initialisée : appeler `init_lightrag`)."""
    from lightrag import LightRAG

    llm = llm or get_llm("indexing", settings)
    embedder = embedder or get_embedder(settings)
    g = settings.graphrag
    settings.paths.lightrag_dir.mkdir(parents=True, exist_ok=True)
    return LightRAG(
        working_dir=str(settings.paths.lightrag_dir),
        llm_model_func=as_lightrag_llm_func(llm),
        llm_model_name=llm.model,
        llm_model_max_async=settings.llm_indexing.max_concurrency,
        embedding_func=as_lightrag_embedding_func(embedder),
        embedding_func_max_async=2,
        embedding_batch_num=settings.embeddings.batch_size,
        # Nos chunks sont déjà découpés (chunking.py). LightRAG ne doit PAS les redécouper : avec
        # 1200 / 100, tout texte de plus de 1100 tokens donnait un 2e petit chunk (le recouvrement),
        # donc un appel LLM de plus par chunk, sans entité. Fenêtre large et sans recouvrement.
        chunk_token_size=settings.chunking.chunk_tokens * 2,
        chunk_overlap_token_size=0,
        entity_extract_max_gleaning=g.entity_extract_max_gleaning,
        force_llm_summary_on_merge=g.force_llm_summary_on_merge,
        enable_llm_cache=g.enable_llm_cache,
        enable_llm_cache_for_entity_extract=g.enable_llm_cache,
        max_parallel_insert=1,
        kg_chunk_pick_method=g.kg_chunk_pick_method,  # voir GraphRAGSettings
        entity_extraction_use_json=True,  # sortie JSON : plus robuste avec les petits modèles
        addon_params={
            "language": g.summary_language,
            "entity_types_guidance": UHA_ENTITY_TYPES_GUIDANCE,
        },
    )


async def init_lightrag(rag) -> None:
    from lightrag.kg.shared_storage import initialize_pipeline_status

    await rag.initialize_storages()
    await initialize_pipeline_status()


def _select(chunks: list[Chunk], limit: int | None) -> list[Chunk]:
    if limit is None:
        return chunks
    docs = list(dict.fromkeys(c.doc_id for c in chunks))[:limit]
    return [c for c in chunks if c.doc_id in docs]


async def _abuild(
    chunks: list[Chunk], settings: Settings, llm: LLMClient, embedder: Embedder
) -> None:
    rag = make_lightrag(settings, llm, embedder)
    await init_lightrag(rag)
    try:
        # Un appel ainsert par chunk : LightRAG saute ceux déjà traités (identifiant = hash du contenu).
        for i, c in enumerate(chunks, start=1):
            log.info("chunk %d/%d  %s", i, len(chunks), c.id)
            await rag.ainsert(c.text, ids=[f"doc-{c.id}"], file_paths=[c.id])
    finally:
        await rag.finalize_storages()


def build_graph(
    chunks: list[Chunk],
    settings: Settings,
    limit: int | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> GraphBuildReport:
    llm = llm or get_llm("indexing", settings)
    embedder = embedder or get_embedder(settings)
    selected = _select(chunks, limit)
    t0 = time.perf_counter()
    llm.reset_usage()
    emb_texts0 = embedder.n_texts

    asyncio.run(_abuild(selected, settings, llm, embedder))

    usage = llm.reset_usage()
    duration = time.perf_counter() - t0
    graph = load_graph(settings)
    meta_path = settings.paths.lightrag_dir / "metadata.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    cumulative = Usage(**previous.get("llm_usage_cumulative", asdict(Usage()))) + usage
    report = GraphBuildReport(
        n_chunks_requested=len(selected),
        n_chunks_total=len(chunks),
        n_entities=graph.number_of_nodes(),
        n_relations=graph.number_of_edges(),
        usage=usage,
        duration_s=round(duration, 1),
        model=llm.model,
    )
    write_metadata(
        meta_path,
        {
            "kind": "graph_index",
            "indexing_model": llm.model,
            "indexing_provider": settings.llm_indexing.provider,
            "embedding_model": embedder.model,
            "chunking": settings.chunking.model_dump(),
            "graphrag": settings.graphrag.model_dump(),
            "n_chunks_inserted_this_run": len(selected),
            "n_entities": report.n_entities,
            "n_relations": report.n_relations,
            "llm_usage_this_run": asdict(usage),
            "llm_usage_cumulative": asdict(cumulative),
            "embedding_texts_this_run": embedder.n_texts - emb_texts0,
            "duration_s_this_run": report.duration_s,
            "duration_s_cumulative": round(previous.get("duration_s_cumulative", 0) + duration, 1),
        },
    )
    log.info(
        "Graphe : %d entités, %d relations ; %d tokens (%d appels) en %.0fs",
        report.n_entities,
        report.n_relations,
        usage.total_tokens,
        usage.n_calls,
        duration,
    )
    return report


def load_graph(settings: Settings) -> nx.Graph:
    path: Path = settings.paths.lightrag_dir / GRAPHML_NAME
    if not path.exists():
        return nx.Graph()
    return nx.read_graphml(path)


def graph_stats(graph: nx.Graph) -> dict:
    from collections import Counter

    types = Counter(d.get("entity_type", "?") for _, d in graph.nodes(data=True))
    degrees = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    return {
        "n_entities": graph.number_of_nodes(),
        "n_relations": graph.number_of_edges(),
        "n_components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "by_type": dict(types.most_common()),
        "top_degree": [(n, d) for n, d in degrees[:15]],
    }


def find_duplicate_entities(
    graph: nx.Graph, embedder: Embedder, threshold: float = 0.9
) -> list[tuple[str, str, float]]:
    """Paires d'entités de même type dont les noms sont très proches (doublons FR / EN probables).

    Ne fusionne rien : le nombre de paires est un résultat à présenter (limite du GraphRAG multilingue).
    """

    names = list(graph.nodes)
    if len(names) < 2:
        return []
    types = {n: graph.nodes[n].get("entity_type", "") for n in names}
    vecs = embedder.embed_documents(names)
    sims = vecs @ vecs.T
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if types[names[i]] == types[names[j]] and sims[i, j] >= threshold:
                pairs.append((names[i], names[j], float(sims[i, j])))
    return sorted(pairs, key=lambda p: -p[2])
