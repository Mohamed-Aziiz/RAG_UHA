"""Index vectoriel du RAG classique : chunks → embeddings BGE-M3 → ChromaDB (via LlamaIndex).

Construction : 0 token de LLM (c'est un point de la comparaison des coûts d'indexation).
La collection est persistante dans `paths.chroma_dir` ; ne jamais l'ouvrir dans un dossier
Drive synchronisé.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import chromadb

from rag_uha.config import Settings
from rag_uha.embeddings import Embedder, as_llamaindex_embedding, get_embedder
from rag_uha.types import Chunk

log = logging.getLogger(__name__)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_metadata(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        **data,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _collection(settings: Settings, reset: bool = False):
    client = chromadb.PersistentClient(path=str(settings.paths.chroma_dir))
    if reset:
        if any(c.name == settings.rag.collection for c in client.list_collections()):
            client.delete_collection(settings.rag.collection)
    return client.get_or_create_collection(
        settings.rag.collection, metadata={"hnsw:space": "cosine"}
    )


def build_vector_index(
    chunks: list[Chunk], settings: Settings, embedder: Embedder | None = None
) -> int:
    """(Re)construit la collection ChromaDB à partir des chunks. Renvoie le nombre de chunks indexés."""
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.chroma import ChromaVectorStore

    embedder = embedder or get_embedder(settings)
    t0 = time.perf_counter()
    collection = _collection(settings, reset=True)
    store = ChromaVectorStore(chroma_collection=collection)
    nodes = [
        TextNode(
            id_=c.id,
            text=c.text,
            metadata={"doc_id": c.doc_id, "lang": c.lang, "position": c.position},
            excluded_embed_metadata_keys=["doc_id", "lang", "position"],
            excluded_llm_metadata_keys=["doc_id", "lang", "position"],
        )
        for c in chunks
    ]
    VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=as_llamaindex_embedding(embedder),
        show_progress=True,
    )
    n = collection.count()
    duration = time.perf_counter() - t0
    write_metadata(
        settings.paths.chroma_dir / "metadata.json",
        {
            "kind": "vector_index",
            "collection": settings.rag.collection,
            "n_chunks": n,
            "embedding_model": embedder.model,
            "embedding_provider": settings.embeddings.provider,
            "embedding_texts": embedder.n_texts,
            "embedding_tokens_est": embedder.n_tokens_est,
            "chunking": settings.chunking.model_dump(),
            "llm_tokens": 0,
            "duration_s": round(duration, 1),
        },
    )
    log.info("Index ChromaDB : %d chunks en %.1fs", n, duration)
    return n


def open_vector_index(settings: Settings, embedder: Embedder | None = None):
    """Ouvre l'index existant (lecture) et renvoie un `VectorStoreIndex` LlamaIndex."""
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    embedder = embedder or get_embedder(settings)
    collection = _collection(settings)
    if collection.count() == 0:
        raise RuntimeError(
            f"Collection vide dans {settings.paths.chroma_dir} : lance scripts/build_vector_index.py "
            "ou scripts/unpack_artifacts.py"
        )
    return VectorStoreIndex.from_vector_store(
        ChromaVectorStore(chroma_collection=collection),
        embed_model=as_llamaindex_embedding(embedder),
    )
