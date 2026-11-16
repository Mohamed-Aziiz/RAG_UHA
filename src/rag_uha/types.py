"""Types partagés par tout le projet.

Les deux pipelines (RAG et GraphRAG) consomment les mêmes `Chunk` et renvoient
le même `Answer` : l'évaluation et l'interface ne savent pas qui a répondu.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Lang = Literal["fr", "en"]
PipelineName = Literal["rag", "graphrag"]
DocFormat = Literal["txt", "md", "html", "pdf"]


@dataclass
class Document:
    """Un document source, après chargement et nettoyage (avant découpage)."""

    doc_id: str  # nom du fichier sans extension, ex. "master-data-science"
    title: str
    text: str
    lang: Lang
    source: str  # chemin ou URL d'origine
    fmt: DocFormat


@dataclass
class Chunk:
    """Morceau de document. Identique pour le RAG et le GraphRAG."""

    id: str  # stable : "<doc_id>::<position>"
    doc_id: str
    text: str
    lang: Lang
    position: int  # 0, 1, 2… dans le document

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rank: int  # 1 = meilleur
    score: float | None = None


@dataclass
class Usage:
    """Tokens consommés. Additionnable pour cumuler plusieurs appels."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            n_calls=self.n_calls + other.n_calls,
        )


@dataclass
class Answer:
    """Résultat d'une question, quel que soit le pipeline."""

    question: str
    text: str
    pipeline: PipelineName
    contexts: list[RetrievedChunk]  # chunks réellement passés au LLM, dans l'ordre
    sources: list[str]  # doc_id distincts, dans l'ordre d'apparition
    usage: Usage  # tokens de CETTE question (recherche + génération)
    latency_s: float  # temps total
    retrieval_latency_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)  # GraphRAG : entités, relations, mode

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation JSON (pour les runs d'évaluation et le cache de démo)."""
        return {
            "question": self.question,
            "text": self.text,
            "pipeline": self.pipeline,
            "contexts": [
                {"chunk": c.chunk.to_dict(), "rank": c.rank, "score": c.score}
                for c in self.contexts
            ],
            "sources": self.sources,
            "usage": asdict(self.usage),
            "latency_s": self.latency_s,
            "retrieval_latency_s": self.retrieval_latency_s,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Answer:
        return cls(
            question=d["question"],
            text=d["text"],
            pipeline=d["pipeline"],
            contexts=[
                RetrievedChunk(Chunk.from_dict(c["chunk"]), c["rank"], c.get("score"))
                for c in d["contexts"]
            ],
            sources=d["sources"],
            usage=Usage(**d["usage"]),
            latency_s=d["latency_s"],
            retrieval_latency_s=d.get("retrieval_latency_s", 0.0),
            extra=d.get("extra", {}),
        )
