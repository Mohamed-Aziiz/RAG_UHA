"""Étape 4 : exécute un pipeline sur le jeu de questions → data/runs/<pipeline>.jsonl (reprise automatique).

uv run python scripts/run_eval.py --pipeline rag
uv run python scripts/run_eval.py --pipeline graphrag
uv run python scripts/run_eval.py --pipeline graphrag --mode mix --tag mix   # variante
uv run python scripts/run_eval.py --check          # valide questions.jsonl sans rien exécuter
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag_uha.chunking import load_chunks
from rag_uha.config import load_settings
from rag_uha.eval.dataset import load_questions, summarize, validate_questions
from rag_uha.eval.runner import run_pipeline
from rag_uha.pipelines.base import build_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
)
for noisy in ("httpx", "lightrag", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--pipeline", default=None, help="rag | graphrag (défaut : config)")
    p.add_argument("--mode", default=None, help="GraphRAG : local | global | hybrid | mix")
    p.add_argument("--tag", default=None, help="suffixe du fichier de sortie (ex. mix)")
    p.add_argument("--limit", type=int, default=None, help="n premières questions seulement")
    p.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="pause entre questions (s), utile en offre gratuite",
    )
    p.add_argument(
        "--questions", default=None,
        help="fichier de questions (défaut : paths.questions_file ; ex. eval/questions_dev.jsonl)",
    )
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    s = load_settings(args.config)

    qs = load_questions(Path(args.questions) if args.questions else s.paths.questions_file)
    known = (
        {c.doc_id for c in load_chunks(s.paths.chunks_file)}
        if s.paths.chunks_file.exists()
        else None
    )
    issues = validate_questions(qs, known)
    print("Jeu de questions :", json.dumps(summarize(qs), ensure_ascii=False))
    if issues:
        print("Problèmes :\n  " + "\n  ".join(issues))
        if args.check:
            raise SystemExit(1)
    if args.check:
        print("OK")
        return

    name = args.pipeline or s.pipeline
    pipe = build_pipeline(name, s)
    if args.mode and name == "graphrag":
        pipe.mode = args.mode
    tag = args.tag or (args.mode if args.mode else None)
    out = s.paths.runs_dir / (f"{name}_{tag}.jsonl" if tag else f"{name}.jsonl")
    records = run_pipeline(pipe, qs[: args.limit] if args.limit else qs, out, pause_s=args.pause)
    tot = sum(r.answer.usage.total_tokens for r in records)
    print(f"\n{len(records)} réponses dans {out} — {tot} tokens au total")


if __name__ == "__main__":
    main()
