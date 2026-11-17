"""Runner, reprise, métriques — avec un pipeline factice."""

from __future__ import annotations

import math
from pathlib import Path

from rag_uha.config import load_settings
from rag_uha.eval.dataset import load_questions, summarize, validate_questions
from rag_uha.eval.metrics import aggregate, hit_at_k, per_question_scores, reciprocal_rank
from rag_uha.eval.runner import load_run, run_pipeline
from rag_uha.types import Answer, Chunk, RetrievedChunk, Usage


class StubPipeline:
    name = "rag"

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, question):
        return []

    def answer(self, question: str) -> Answer:
        self.calls += 1
        ctx = [
            RetrievedChunk(Chunk("uha-presentation::0", "uha-presentation", "x", "fr", 0), 1, 0.9),
            RetrievedChunk(Chunk("uha-overview::0", "uha-overview", "y", "en", 0), 2, 0.8),
        ]
        text = (
            "Je ne trouve pas cette information dans les documents."
            if "doctorat" in question
            else "Mulhouse et Colmar [uha-presentation]"
        )
        return Answer(
            question,
            text,
            "rag",
            ctx,
            ["uha-presentation", "uha-overview"],
            Usage(100, 20, 1),
            1.5,
            0.3,
            {"model": "openai/gpt-oss-120b"},
        )


def test_metrics_basic():
    assert hit_at_k(["a", "b"], ["b"], 1) == 0.0 and hit_at_k(["a", "b"], ["b"], 2) == 1.0
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5 and reciprocal_rank(["a"], ["z"]) == 0.0
    assert math.isnan(reciprocal_rank(["a"], []))


def test_runner_resume_and_scores(tmp_path: Path):
    s = load_settings()
    qs = load_questions(s.paths.questions_file)
    assert validate_questions(qs, {"uha-presentation", "uha-overview"}) == []
    assert summarize(qs)["by_category"]["hors_corpus"] == 1

    out = tmp_path / "rag.jsonl"
    pipe = StubPipeline()
    recs = run_pipeline(pipe, qs[:3], out)
    assert len(recs) == 3 and pipe.calls == 3
    recs = run_pipeline(pipe, qs, out)  # reprise : seules les 2 restantes sont exécutées
    assert len(recs) == 5 and pipe.calls == 5
    assert [r.question.id for r in load_run(out)] == [q.id for q in qs]

    df = per_question_scores(recs, k=s.eval.k, prices=s.eval.price_per_m_tokens)
    assert df.loc[df.id == "q01", "hit_at_k"].item() == 1.0
    assert df.loc[df.id == "q02", "mrr"].item() == 0.5
    assert df.loc[df.id == "q05", "correct_refusal"].item()
    agg = aggregate(df)
    assert (
        agg["n_questions"] == 5
        and agg["correct_refusal_rate"] == 1.0
        and agg["false_refusal_rate"] == 0.0
    )
    assert agg["cost_usd_per_question"] > 0
