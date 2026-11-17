"""Tableau comparatif final et figures (CSV + Markdown + PNG dans eval/results/)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rag_uha.config import Settings
from rag_uha.eval.dataset import RunRecord
from rag_uha.eval.metrics import aggregate, per_question_scores

LABELS = {
    "answer_relevancy": ("Pertinence", "Answer relevancy (RAGAS) ↑"),
    "answer_correctness": ("Pertinence", "Answer correctness vs référence (RAGAS) ↑"),
    "faithfulness": ("Fiabilité", "Faithfulness / fidélité aux sources (RAGAS) ↑"),
    "correct_refusal_rate": ("Fiabilité", "Refus corrects sur questions hors corpus ↑"),
    "false_refusal_rate": ("Fiabilité", "Refus à tort sur questions répondables ↓"),
    "hit_at_k": ("Récupération", "Hit@k (niveau document) ↑"),
    "mrr": ("Récupération", "MRR (niveau document) ↑"),
    "doc_recall_at_k": ("Récupération", "Rappel des documents attendus @k ↑"),
    "context_precision": ("Récupération", "Context precision (RAGAS) ↑"),
    "latency_median_s": ("Performances", "Latence médiane par question (s) ↓"),
    "latency_p95_s": ("Performances", "Latence p95 (s) ↓"),
    "retrieval_latency_median_s": ("Performances", "Latence de recherche médiane (s) ↓"),
    "tokens_mean": ("Performances", "Tokens LLM par question ↓"),
    "calls_mean": ("Performances", "Appels LLM par question ↓"),
    "cost_usd_per_question": ("Performances", "Coût estimé par question ($) ↓"),
}


def build_tables(
    runs: dict[str, list[RunRecord]], ragas: dict[str, pd.DataFrame], settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Renvoie (par question, principal, par catégorie, par langue)."""
    frames = []
    for name, records in runs.items():
        df = per_question_scores(records, settings.eval.k, settings.eval.price_per_m_tokens)
        df["pipeline"] = name
        if name in ragas and len(ragas[name]):
            df = df.merge(ragas[name], on="id", how="left")
        frames.append(df)
    per_q = pd.concat(frames, ignore_index=True)

    main = pd.DataFrame({name: aggregate(g) for name, g in per_q.groupby("pipeline", sort=False)})
    main = main.loc[[k for k in LABELS if k in main.index]]
    main.insert(0, "axe", [LABELS[k][0] for k in main.index])
    main.insert(1, "métrique", [LABELS[k][1] for k in main.index])

    def _by(col: str) -> pd.DataFrame:
        rows = {}
        for (name, val), g in per_q.groupby(["pipeline", col], sort=False):
            rows[(name, val)] = aggregate(g)
        t = pd.DataFrame(rows).T
        keep = [
            c
            for c in (
                "n_questions",
                "hit_at_k",
                "mrr",
                "faithfulness",
                "answer_correctness",
                "latency_median_s",
                "tokens_mean",
            )
            if c in t.columns
        ]
        return t[keep]

    return per_q, main, _by("category"), _by("lang")


def indexing_table(settings: Settings) -> pd.DataFrame:
    rows = {}
    for name, d in (("rag", settings.paths.chroma_dir), ("graphrag", settings.paths.lightrag_dir)):
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        size_mb = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e6
        usage = m.get("llm_usage_cumulative", {})
        model = m.get("indexing_model", "")
        pin, pout = settings.eval.price_per_m_tokens.get(model, (0.0, 0.0))
        rows[name] = {
            "modèle LLM d'indexation": model or "— (aucun)",
            "tokens LLM": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            "appels LLM": usage.get("n_calls", 0),
            "coût LLM estimé ($)": round(
                (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout)
                / 1e6,
                4,
            ),
            "durée (s)": m.get("duration_s_cumulative", m.get("duration_s", 0)),
            "entités": m.get("n_entities", "—"),
            "relations": m.get("n_relations", "—"),
            "taille sur disque (Mo)": round(size_mb, 1),
        }
    return pd.DataFrame(rows)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "—" if pd.isna(v) else (f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}")
    return str(v)


def to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.values) + " |")
    return "\n".join(lines)


def save_figures(per_q: pd.DataFrame, main: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    pipelines = [c for c in main.columns if c not in ("axe", "métrique")]
    colors = {"rag": "#4C72B0", "graphrag": "#DD8452"}

    # 1. barres : métriques de qualité (0-1)
    quality = [
        k
        for k in ("answer_correctness", "answer_relevancy", "faithfulness", "hit_at_k", "mrr")
        if k in main.index
    ]
    if quality:
        fig, ax = plt.subplots(figsize=(8, 4))
        w = 0.8 / len(pipelines)
        for i, p in enumerate(pipelines):
            vals = [float(main.loc[k, p]) for k in quality]
            ax.bar([x + i * w for x in range(len(quality))], vals, w, label=p, color=colors.get(p))
        ax.set_xticks([x + w * (len(pipelines) - 1) / 2 for x in range(len(quality))])
        ax.set_xticklabels(quality, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_title("Qualité (moyenne sur les questions)")
        ax.legend()
        fig.tight_layout()
        paths.append(out_dir / "fig_quality.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

    # 2. latence et tokens
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, title in (
        (axes[0], "latency_s", "Latence par question (s)"),
        (axes[1], "total_tokens", "Tokens LLM par question"),
    ):
        data = [per_q.loc[per_q.pipeline == p, col].dropna() for p in pipelines]
        ax.boxplot(data, tick_labels=pipelines)
        ax.set_title(title)
    fig.tight_layout()
    paths.append(out_dir / "fig_perf.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # 3. par catégorie de question
    metric = "answer_correctness" if "answer_correctness" in per_q.columns else "hit_at_k"
    cats = [c for c in ("factual", "multi_hop", "inter_langues") if c in set(per_q.category)]
    if cats:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        w = 0.8 / len(pipelines)
        for i, p in enumerate(pipelines):
            vals = [
                per_q[(per_q.pipeline == p) & (per_q.category == c)][metric].mean() for c in cats
            ]
            ax.bar([x + i * w for x in range(len(cats))], vals, w, label=p, color=colors.get(p))
        ax.set_xticks([x + w * (len(pipelines) - 1) / 2 for x in range(len(cats))])
        ax.set_xticklabels(cats)
        ax.set_ylim(0, 1)
        ax.set_title(f"{metric} par type de question")
        ax.legend()
        fig.tight_layout()
        paths.append(out_dir / "fig_by_category.png")
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


def examples_markdown(runs: dict[str, list[RunRecord]], per_q: pd.DataFrame, n: int = 3) -> str:
    """Quelques questions où les deux pipelines divergent le plus (à commenter à la main dans le README)."""
    if "answer_correctness" not in per_q.columns or len(runs) < 2:
        return ""
    piv = per_q.pivot_table(index="id", columns="pipeline", values="answer_correctness")
    if piv.shape[1] < 2:
        return ""
    a, b = piv.columns[:2]
    piv["diff"] = piv[b] - piv[a]
    picks = list(piv.sort_values("diff").index[:n]) + list(piv.sort_values("diff").index[-n:])
    by_id = {name: {r.question.id: r for r in recs} for name, recs in runs.items()}
    out = []
    for qid in dict.fromkeys(picks):
        q = next(iter(by_id.values()))[qid].question
        out.append(
            f"### {qid} ({q.category}, {q.lang}) — {q.question}\n\n*Référence :* {q.reference_answer}\n"
        )
        for name in runs:
            r = by_id[name].get(qid)
            if r:
                sc = piv.loc[qid, name]
                out.append(
                    f"**{name}** (correctness {sc:.2f}, sources {', '.join(r.answer.sources[:4])}) :\n> {r.answer.text[:600]}\n"
                )
    return "\n".join(out)
