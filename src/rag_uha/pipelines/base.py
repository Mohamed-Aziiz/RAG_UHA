"""Interface commune des pipelines et fabrique."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_uha.config import Settings, load_settings
from rag_uha.types import Answer, RetrievedChunk


@runtime_checkable
class QAPipeline(Protocol):
    name: str

    def retrieve(self, question: str) -> list[RetrievedChunk]: ...

    def answer(self, question: str) -> Answer: ...


def build_pipeline(name: str | None = None, settings: Settings | None = None) -> QAPipeline:
    """`name` : "rag" | "graphrag" ; None → `settings.pipeline`."""
    settings = settings or load_settings()
    name = name or settings.pipeline
    if name == "rag":
        from rag_uha.pipelines.rag import RAGPipeline

        return RAGPipeline(settings)
    if name == "graphrag":
        from rag_uha.pipelines.graphrag import GraphRAGPipeline

        return GraphRAGPipeline(settings)
    raise ValueError(f"Pipeline inconnu : {name!r} (attendu : rag | graphrag)")
