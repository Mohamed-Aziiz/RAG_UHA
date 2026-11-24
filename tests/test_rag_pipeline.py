"""Index ChromaDB réel (dans un dossier temporaire) + pipeline RAG avec doublures."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_uha.config import load_settings
from rag_uha.indexing.vector_index import build_vector_index
from rag_uha.pipelines import rag as rag_module
from rag_uha.pipelines.rag import RAGPipeline
from rag_uha.types import Chunk

CHUNKS = [
    Chunk(
        "master-ds::0",
        "master-ds",
        "Le Master Data Science forme des spécialistes de l'apprentissage automatique et des données.",
        "fr",
        0,
    ),
    Chunk(
        "master-ds::1",
        "master-ds",
        "L'admission au Master Data Science se fait sur dossier via Mon Master.",
        "fr",
        1,
    ),
    Chunk(
        "housing::0",
        "housing",
        "The international relations office helps students with housing and visas.",
        "en",
        0,
    ),
    Chunk(
        "library::0",
        "library",
        "La bibliothèque universitaire est ouverte du lundi au samedi.",
        "fr",
        0,
    ),
]


@pytest.fixture
def settings(tmp_path: Path):
    s = load_settings()
    s.paths.chroma_dir = tmp_path / "chroma"
    s.rag.rerank = False  # les tests tournent hors ligne : pas d'appel au reranker
    s.rag.hybrid = False  # et sur leurs propres chunks, pas sur data/chunks
    return s


def test_build_and_query(settings, hash_embedder, fake_llm, monkeypatch):
    n = build_vector_index(CHUNKS, settings, embedder=hash_embedder)
    assert n == 4
    assert (settings.paths.chroma_dir / "metadata.json").exists()

    monkeypatch.setattr(rag_module, "get_embedder", lambda s: hash_embedder)
    pipe = RAGPipeline(settings, llm=fake_llm)

    hits = pipe.retrieve("admission Master Data Science dossier", top_k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].chunk.doc_id == "master-ds"
    assert hits[0].chunk.id in {"master-ds::0", "master-ds::1"}

    ans = pipe.answer("Comment se fait l'admission au Master Data Science ?")
    assert ans.pipeline == "rag" and ans.text == fake_llm.reply
    assert ans.sources[0] == "master-ds"
    assert len(ans.contexts) == settings.rag.top_k
    assert ans.usage.n_calls == 1 and ans.latency_s >= ans.retrieval_latency_s
    system, prompt = fake_llm.prompts[-1]
    assert "SAME LANGUAGE" in system and "[source: master-ds]" in prompt


def test_rebuild_resets_collection(settings, hash_embedder):
    build_vector_index(CHUNKS, settings, embedder=hash_embedder)
    assert build_vector_index(CHUNKS[:2], settings, embedder=hash_embedder) == 2
