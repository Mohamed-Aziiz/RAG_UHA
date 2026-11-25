"""Métriques maison : récupération (Hit@k, MRR), fiabilité (refus), performances (latence, tokens, coût)."""

from __future__ import annotations

import pandas as pd

from rag_uha.eval.dataset import RunRecord
from rag_uha.prompts import is_refusal


# ------------------------------------------------------------------ récupération (niveau document)
def retrieved_docs(record: RunRecord) -> list[str]:
    """Documents des contextes, dans l'ordre de rang, sans doublon."""
    return list(
        dict.fromkeys(
            rc.chunk.doc_id for rc in sorted(record.answer.contexts, key=lambda r: r.rank)
        )
    )


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    return float(any(d in gold for d in retrieved[:k])) if gold else float("nan")


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return float("nan")
    for i, d in enumerate(retrieved, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def doc_recall(retrieved: list[str], gold: list[str], k: int) -> float:
    """Part des documents attendus présents dans le top-k (utile pour les multi-sauts)."""
    if not gold:
        return float("nan")
    return sum(1 for g in gold if g in retrieved[:k]) / len(gold)


# ------------------------------------------------------------------ coût
def cost_usd(
    prompt_tokens: int, completion_tokens: int, model: str, prices: dict[str, tuple[float, float]]
) -> float:
    pin, pout = prices.get(model, (0.0, 0.0))
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000


# ------------------------------------------------------------------ tableau par question
def per_question_scores(
    records: list[RunRecord], k: int, prices: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    rows = []
    for r in records:
        q, a = r.question, r.answer
        ret = retrieved_docs(r)
        model = a.extra.get("model", "")
        rows.append(
            {
                "id": q.id,
                "pipeline": a.pipeline,
                "category": q.category,
                "lang": q.lang,
                "answerable": q.answerable,
                "hit_at_k": hit_at_k(ret, q.gold_docs, k),
                "mrr": reciprocal_rank(ret, q.gold_docs),
                "doc_recall_at_k": doc_recall(ret, q.gold_docs, k),
                "n_contexts": len(a.contexts),
                "refused": is_refusal(a.text),
                "correct_refusal": (not q.answerable) and is_refusal(a.text),
                "false_refusal": q.answerable and is_refusal(a.text),
                "latency_s": a.latency_s,
                "retrieval_latency_s": a.retrieval_latency_s,
                "prompt_tokens": a.usage.prompt_tokens,
                "completion_tokens": a.usage.completion_tokens,
                "total_tokens": a.usage.total_tokens,
                "n_calls": a.usage.n_calls,
                "cost_usd": cost_usd(
                    a.usage.prompt_tokens, a.usage.completion_tokens, model, prices
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.Series:
    """Une ligne de résumé pour un pipeline (moyennes, médianes, p95)."""
    ans = df[df["answerable"]]
    out = {
        "n_questions": len(df),
        "hit_at_k": ans["hit_at_k"].mean(),
        "mrr": ans["mrr"].mean(),
        "doc_recall_at_k": ans["doc_recall_at_k"].mean(),
        "correct_refusal_rate": df.loc[~df["answerable"], "correct_refusal"].mean()
        if (~df["answerable"]).any()
        else float("nan"),
        "false_refusal_rate": ans["false_refusal"].mean(),
        "latency_median_s": df["latency_s"].median(),
        "latency_p95_s": df["latency_s"].quantile(0.95),
        "retrieval_latency_median_s": df["retrieval_latency_s"].median(),
        "tokens_mean": df["total_tokens"].mean(),
        "prompt_tokens_mean": df["prompt_tokens"].mean(),
        "calls_mean": df["n_calls"].mean(),
        "cost_usd_per_question": df["cost_usd"].mean(),
        "cost_usd_total": df["cost_usd"].sum(),
    }
    # Scores RAGAS : moyenne sur les questions REPONDABLES seulement. Sur une question hors corpus,
    # la bonne reponse est un refus, que RAGAS note 0 en pertinence ("non engagee") : l'inclure
    # penaliserait le comportement attendu. Les hors corpus sont evaluees par correct_refusal_rate.
    for col in ("faithfulness", "answer_relevancy", "answer_correctness", "context_precision"):
        if col in df.columns:
            out[col] = ans[col].mean()
    # Fidelite : mesure les hallucinations, donc seulement sur les reponses qui AFFIRMENT quelque
    # chose. Un refus n'a aucune affirmation a verifier : RAGAS lui donne 0 ou 1 au hasard
    # (constate sur la validation : R01 -> 0,0, R10 -> 1,0). Les refus sont comptes a part.
    if "faithfulness" in df.columns:
        out["faithfulness"] = ans.loc[~ans["refused"], "faithfulness"].mean()
    return pd.Series(out)
