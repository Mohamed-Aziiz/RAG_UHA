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
    # Jeu d'exemple propre au test : ne depend pas de eval/questions.jsonl, qui evolue
    # au fil du projet (les 25 vraies questions ont des gold_docs reels).
    qs = load_questions(Path(__file__).parent / "fixtures" / "questions_example.jsonl")
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


def test_report_tables_and_figures(tmp_path: Path):
    import pandas as pd

    from rag_uha.eval.report import build_tables, examples_markdown, save_figures, to_markdown

    s = load_settings()
    qs = load_questions(s.paths.questions_file)
    recs = run_pipeline(StubPipeline(), qs, tmp_path / "rag.jsonl")
    import copy

    g_recs = copy.deepcopy(recs)
    for r in g_recs:
        r.answer.pipeline = "graphrag"
        r.answer.latency_s *= 2
    ragas = {
        "rag": pd.DataFrame(
            {"id": [q.id for q in qs], "faithfulness": 0.8, "answer_correctness": 0.6}
        ),
        "graphrag": pd.DataFrame(
            {"id": [q.id for q in qs], "faithfulness": 0.9, "answer_correctness": 0.7}
        ),
    }
    per_q, main, _by_cat, _by_lang = build_tables({"rag": recs, "graphrag": g_recs}, ragas, s)
    assert set(main.columns) == {"axe", "métrique", "rag", "graphrag"}
    assert abs(main.loc["faithfulness", "graphrag"] - 0.9) < 1e-9
    assert main.loc["latency_median_s", "graphrag"] == 2 * main.loc["latency_median_s", "rag"]
    assert "| axe |" in to_markdown(main.reset_index(drop=True), index=False)
    figs = save_figures(per_q, main, tmp_path / "res")
    assert len(figs) == 3 and all(f.exists() for f in figs)
    assert "q0" in examples_markdown({"rag": recs, "graphrag": g_recs}, per_q)


def test_ragas_means_exclude_out_of_corpus():
    import pandas as pd

    df = pd.DataFrame(
        {
            "answerable": [True, True, False],
            "hit_at_k": [1.0, 0.0, float("nan")],
            "mrr": [1.0, 0.0, float("nan")],
            "doc_recall_at_k": [1.0, 0.0, float("nan")],
            "correct_refusal": [False, False, True],
            "false_refusal": [False, False, False],
            "latency_s": [1, 1, 1],
            "retrieval_latency_s": [0, 0, 0],
            "total_tokens": [1, 1, 1],
            "prompt_tokens": [1, 1, 1],
            "n_calls": [1, 1, 1],
            "cost_usd": [0, 0, 0],
            "answer_relevancy": [0.8, 0.6, 0.0],
        }
    )
    assert aggregate(df)["answer_relevancy"] == 0.7
