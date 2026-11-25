"""Indexation LightRAG réelle (dossier temporaire) avec un LLM factice qui imite l'extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rag_uha.config import load_settings
from rag_uha.indexing.graph_index import (
    build_graph,
    find_duplicate_entities,
    graph_stats,
    load_graph,
)
from rag_uha.pipelines import graphrag as gr_module
from rag_uha.pipelines.graphrag import GraphRAGPipeline
from rag_uha.types import Chunk
from rag_uha.viz import to_html, top_subgraph
from tests.conftest import FakeLLM

CHUNKS = [
    Chunk(
        "master-ds::0",
        "master-ds",
        "Le Master Data Science est proposé par la Faculté des Sciences et Techniques (FST) de l'Université de Haute-Alsace (UHA).",
        "fr",
        0,
    ),
    Chunk(
        "fst::0",
        "fst",
        "The Faculté des Sciences et Techniques (FST) is located on the Illberg Campus in Mulhouse.",
        "en",
        0,
    ),
]


class FakeGraphLLM(FakeLLM):
    """Imite LightRAG : mots-clés JSON, extraction JSON des groupes de mots capitalisés, sinon texte."""

    def complete(self, prompt, system=None, history=None, max_tokens=None, **kw):
        self.prompts.append((system, prompt))
        if "high_level_keywords" in prompt or "high_level_keywords" in (system or ""):
            out = json.dumps(
                {
                    "high_level_keywords": ["university programs"],
                    "low_level_keywords": ["Master Data Science", "FST"],
                }
            )
        elif "Input Text" in prompt or "entities" in (system or ""):
            text = prompt.split("---Input Text---")[-1] if "---Input Text---" in prompt else prompt
            names = sorted(
                set(
                    re.findall(
                        r"(?:[A-ZÉ][\w'’-]+(?: (?:de |des |du |et |d')?[A-ZÉ][\w'’-]+)*(?: \([A-Z]+\))?)",
                        text,
                    )
                )
            )
            names = [n for n in names if len(n) > 3][:6]
            ents = [
                {"name": n, "type": "Composante", "description": f"{n} is mentioned in the text."}
                for n in names
            ]
            rels = [
                {
                    "source": names[i],
                    "target": names[i + 1],
                    "keywords": "related",
                    "description": "appear together",
                }
                for i in range(len(names) - 1)
            ]
            out = json.dumps({"entities": ents, "relationships": rels})
        else:
            out = "Réponse factice [master-ds]"
        from rag_uha.types import Usage

        self.last_usage = Usage(len(prompt) // 4, len(out) // 4, 1)
        self.usage = self.usage + self.last_usage
        return out


@pytest.fixture
def settings(tmp_path: Path):
    s = load_settings()
    s.paths.lightrag_dir = tmp_path / "lightrag"
    s.paths.chunks_file = tmp_path / "chunks.jsonl"
    s.paths.chunks_file.write_text(
        "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in CHUNKS), encoding="utf-8"
    )
    return s


def test_build_graph_and_query(settings, hash_embedder, monkeypatch, tmp_path):
    llm = FakeGraphLLM()
    report = build_graph(CHUNKS, settings, llm=llm, embedder=hash_embedder)
    assert report.n_chunks_requested == 2 and report.n_entities >= 2 and report.usage.n_calls >= 2
    meta = json.loads((settings.paths.lightrag_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["indexing_model"] == "fake" and meta["n_entities"] == report.n_entities

    g = load_graph(settings)
    st = graph_stats(g)
    assert st["n_entities"] == report.n_entities and st["by_type"]
    pairs = find_duplicate_entities(g, hash_embedder, threshold=0.99)
    assert isinstance(pairs, list)
    html = to_html(top_subgraph(g), tmp_path / "g.html")
    assert html.exists() and html.stat().st_size > 1000

    # reprise : relancer ne refait pas l'extraction (documents déjà connus)
    report2 = build_graph(CHUNKS, settings, llm=llm, embedder=hash_embedder)
    assert report2.n_entities == report.n_entities
    assert report2.usage.n_calls == 0  # aucun appel LLM : documents déjà connus
    meta = json.loads((settings.paths.lightrag_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["llm_usage_cumulative"]["n_calls"] == report.usage.n_calls

    monkeypatch.setattr(gr_module, "get_embedder", lambda s: hash_embedder)
    pipe = GraphRAGPipeline(settings, llm=llm)
    ans = pipe.answer("Où se trouve la FST ?")
    assert ans.pipeline == "graphrag" and ans.text.startswith("Réponse factice")
    assert ans.usage.n_calls >= 2  # mots-clés + réponse
    assert ans.extra["mode"] == "hybrid" and "entities" in ans.extra
    assert ans.extra["keywords_reequilibrage"]["ll_brut"]  # le chemin du correctif a bien été emprunté
    assert all("::" in rc.chunk.id for rc in ans.contexts)
    assert set(ans.sources) <= {"master-ds", "fst"}
    system, prompt = llm.prompts[-1]
    assert "SAME LANGUAGE" in system and "Knowledge graph" in prompt


def test_lightrag_does_not_resplit_our_chunks(settings, hash_embedder):
    """Un chunk de 1 200 tokens doit donner UN seul chunk LightRAG (donc un seul appel d'extraction)."""
    from rag_uha.indexing.graph_index import make_lightrag

    rag = make_lightrag(settings, llm=FakeGraphLLM(), embedder=hash_embedder)
    assert rag.chunk_overlap_token_size == 0
    assert rag.chunk_token_size >= settings.chunking.chunk_tokens + settings.chunking.overlap_tokens


def test_judge_sees_graph_context():
    """Le juge reçoit les passages ET le résumé du graphe envoyé au LLM."""
    import asyncio

    from rag_uha.eval.dataset import EvalQuestion, RunRecord
    from rag_uha.eval.ragas_eval import _score_one
    from rag_uha.types import Answer, RetrievedChunk, Usage

    seen = {}

    class M:
        async def ascore(self, **kw):
            seen.update(kw)
            return 1.0

    ans = Answer(
        "q",
        "r",
        "graphrag",
        [RetrievedChunk(CHUNKS[0], 1)],
        ["master-ds"],
        Usage(),
        1.0,
        extra={"graph_context": "Knowledge graph — entities:\n- FST (Composante): ..."},
    )
    q = EvalQuestion("q1", "q", "fr", "factual", "ref", ["master-ds"])
    asyncio.run(_score_one(RunRecord(q, ans), {"faithfulness": M()}))
    assert seen["retrieved_contexts"][-1].startswith("Knowledge graph")
    assert len(seen["retrieved_contexts"]) == 2


def test_generic_keywords_are_rebalanced(settings):
    """Un mot-clé bas niveau présent partout passe en haut niveau ; un bas niveau vide est rempli."""
    from rag_uha.pipelines.graphrag import _norme

    p = GraphRAGPipeline.__new__(GraphRAGPipeline)
    p.settings = settings
    p.settings.graphrag.generic_keyword_share = 0.5
    p._textes = [
        _norme(t)
        for t in (
            "L'UHA propose des formations.",
            "UHA : vie étudiante.",
            "Campus de l'UHA à Mulhouse.",
            "Master Ingénierie mathématique et data science.",
        )
    ]
    p._parts = {}
    hl, ll = p._reequilibrer(["data science", "master program"], ["UHA"])
    assert "UHA" in hl and "UHA" not in ll
    assert ll == ["data science", "master program"] or "data science" in ll
    hl, ll = p._reequilibrer(["x"], ["Mulhouse", "UHA"])
    assert ll == ["Mulhouse"] and "UHA" in hl
    p.settings.graphrag.generic_keyword_share = 0
    assert p._reequilibrer(["a"], ["UHA"]) == (["a"], ["UHA"])
