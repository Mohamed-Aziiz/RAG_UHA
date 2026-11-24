"""Juge RAGAS (API « collections », RAGAS 0.4) : faithfulness, answer_relevancy, answer_correctness,
context_precision (optionnel).

Le juge est `llm_judge` (qwen/qwen3.8-27b par défaut) : différent du générateur, pour éviter
l'auto-évaluation. Les scores sont écrits question par question dans un JSONL, avec reprise.

    scores = judge_records(records, settings, out_file)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path

import pandas as pd

from rag_uha.config import Settings
from rag_uha.embeddings import as_ragas_embeddings, get_embedder
from rag_uha.eval.dataset import RunRecord
from rag_uha.llm import as_ragas_llm, get_llm

log = logging.getLogger(__name__)


def _metrics(settings: Settings, names: list[str]):
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecisionWithReference,
        Faithfulness,
    )

    judge = as_ragas_llm(get_llm("judge", settings))
    emb = as_ragas_embeddings(get_embedder(settings))
    available = {
        "faithfulness": lambda: Faithfulness(llm=judge),
        "answer_relevancy": lambda: AnswerRelevancy(llm=judge, embeddings=emb, strictness=1),
        "answer_correctness": lambda: AnswerCorrectness(llm=judge, embeddings=emb),
        "context_precision": lambda: ContextPrecisionWithReference(llm=judge),
    }
    return {n: available[n]() for n in names if n in available}


def _value(res) -> float:
    v = getattr(res, "value", res)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


async def _score_one(rec: RunRecord, metrics: dict) -> dict[str, float]:
    q, a = rec.question, rec.answer
    # Contextes = TOUT ce que le LLM de generation a recu : les passages, plus, pour le GraphRAG,
    # le resume des entites et relations du graphe (meme regle pour les deux pipelines).
    contexts = [rc.chunk.text for rc in a.contexts]
    if a.extra.get("graph_context"):
        contexts.append(a.extra["graph_context"])
    contexts = contexts or ["(no context retrieved)"]
    out: dict[str, float] = {}
    for name, m in metrics.items():
        try:
            if name == "faithfulness":
                r = await m.ascore(
                    user_input=q.question, response=a.text, retrieved_contexts=contexts
                )
            elif name == "answer_relevancy":
                r = await m.ascore(user_input=q.question, response=a.text)
            elif name == "answer_correctness":
                r = await m.ascore(
                    user_input=q.question, response=a.text, reference=q.reference_answer
                )
            elif name == "context_precision":
                r = await m.ascore(
                    user_input=q.question, reference=q.reference_answer, retrieved_contexts=contexts
                )
            else:
                continue
            out[name] = _value(r)
        except Exception as exc:  # noqa: BLE001 — un échec de parsing ne doit pas arrêter le run
            log.warning("%s / %s : %s", q.id, name, exc)
            out[name] = float("nan")
    return out


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: {k: v for k, v in r.items() if k != "id"} for r in rows}


def judge_records(
    records: list[RunRecord],
    settings: Settings,
    out_file: Path,
    metric_names: list[str] | None = None,
) -> pd.DataFrame:
    names = metric_names or settings.eval.ragas_metrics
    metrics = _metrics(settings, names)
    done = load_scores(out_file)
    todo = [
        r
        for r in records
        if r.question.id not in done
        or any(math.isnan(done[r.question.id].get(n, float("nan"))) for n in names)
    ]
    log.info(
        "Juge %s : %d questions à noter (%d déjà faites)",
        settings.llm_judge.model,
        len(todo),
        len(done),
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    judge_llm = get_llm("judge", settings)
    with out_file.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, start=1):
            t0 = time.perf_counter()
            scores = asyncio.run(_score_one(rec, metrics))
            done[rec.question.id] = scores
            f.write(json.dumps({"id": rec.question.id, **scores}, ensure_ascii=False) + "\n")
            f.flush()
            log.info(
                "[%d/%d] %s %s (%.0fs)",
                i,
                len(todo),
                rec.question.id,
                {k: round(v, 2) if not math.isnan(v) else None for k, v in scores.items()},
                time.perf_counter() - t0,
            )
    log.info("Juge : %d tokens consommés au total", judge_llm.usage.total_tokens)

    df = pd.DataFrame([{"id": rid, **sc} for rid, sc in done.items()])
    return df[df["id"].isin({r.question.id for r in records})].reset_index(drop=True)
