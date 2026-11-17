"""Étape 5 : note les réponses d'un run avec RAGAS → data/runs/<run>_ragas.jsonl (reprise automatique).

uv run python scripts/judge.py --run rag --limit 2        # test sur 2 questions d'abord
uv run python scripts/judge.py --run rag
uv run python scripts/judge.py --run graphrag
uv run python scripts/judge.py --run graphrag --metrics faithfulness answer_correctness
"""

from __future__ import annotations

import argparse
import logging

from rag_uha.config import load_settings
from rag_uha.eval.ragas_eval import judge_records
from rag_uha.eval.runner import load_run

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
)
for noisy in ("httpx", "ragas", "instructor"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--run", required=True, help="nom du run : rag | graphrag | graphrag_mix …")
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    s = load_settings(args.config)

    records = load_run(s.paths.runs_dir / f"{args.run}.jsonl")
    if not records:
        raise SystemExit(
            f"Aucun run {args.run} dans {s.paths.runs_dir} : lance scripts/run_eval.py d'abord"
        )
    if args.limit:
        records = records[: args.limit]
    df = judge_records(records, s, s.paths.runs_dir / f"{args.run}_ragas.jsonl", args.metrics)
    print(df.describe().loc[["count", "mean", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
