"""Étape 6 : tableaux comparatifs et figures → eval/results/ (à coller dans le README).

uv run python scripts/make_report.py
uv run python scripts/make_report.py --runs rag graphrag graphrag_mix
"""

from __future__ import annotations

import argparse

from rag_uha.config import load_settings
from rag_uha.eval.ragas_eval import load_scores
from rag_uha.eval.report import (
    build_tables,
    examples_markdown,
    indexing_table,
    save_figures,
    to_markdown,
)
from rag_uha.eval.runner import load_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--runs", nargs="+", default=["rag", "graphrag"])
    args = p.parse_args()
    s = load_settings(args.config)
    out = s.paths.results_dir
    out.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    runs, ragas = {}, {}
    for name in args.runs:
        recs = load_run(s.paths.runs_dir / f"{name}.jsonl")
        if not recs:
            print(f"(run {name} absent, ignoré)")
            continue
        runs[name] = recs
        sc = load_scores(s.paths.runs_dir / f"{name}_ragas.jsonl")
        if sc:
            ragas[name] = pd.DataFrame([{"id": k, **v} for k, v in sc.items()])
    if not runs:
        raise SystemExit("Aucun run trouvé.")

    per_q, main, by_cat, by_lang = build_tables(runs, ragas, s)
    idx = indexing_table(s)
    per_q.to_csv(out / "per_question.csv", index=False)
    main.to_csv(out / "comparison.csv")
    by_cat.to_csv(out / "by_category.csv")
    by_lang.to_csv(out / "by_lang.csv")
    idx.to_csv(out / "indexing.csv")
    figs = save_figures(per_q, main, out)

    md = [
        "## Tableau comparatif principal\n",
        to_markdown(main.reset_index(drop=True), index=False),
        "\n\n## Par type de question\n",
        to_markdown(by_cat),
        "\n\n## Par langue de la question\n",
        to_markdown(by_lang),
        "\n\n## Coût d'indexation\n",
        to_markdown(idx),
        "\n\n## Figures\n",
        "\n".join(f"![{f.stem}]({f.name})" for f in figs),
        "\n\n## Exemples où les pipelines divergent (à commenter)\n",
        examples_markdown(runs, per_q),
    ]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(to_markdown(main.reset_index(drop=True), index=False))
    print(f"\n→ {out / 'results.md'} + CSV + {len(figs)} figures")


if __name__ == "__main__":
    main()
