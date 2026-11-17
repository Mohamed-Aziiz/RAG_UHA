"""Pose une question à un pipeline et affiche réponse, sources, tokens, latence.

uv run python scripts/ask.py "Quelles formations propose la FST ?"
uv run python scripts/ask.py --pipeline graphrag "What is the Master in Data Science?"
uv run python scripts/ask.py --pipeline both "..."
"""

from __future__ import annotations

import argparse
import logging

from rag_uha.config import load_settings
from rag_uha.pipelines.base import build_pipeline

logging.basicConfig(level=logging.WARNING)


def show(ans) -> None:
    print(
        f"\n=== {ans.pipeline.upper()}  ({ans.latency_s:.1f}s dont recherche {ans.retrieval_latency_s:.1f}s, "
        f"{ans.usage.prompt_tokens}+{ans.usage.completion_tokens} tokens, {ans.usage.n_calls} appels)"
    )
    print(ans.text)
    print("\nSources :", ", ".join(ans.sources) or "—")
    for rc in ans.contexts:
        score = f" score={rc.score:.3f}" if rc.score is not None else ""
        print(
            f"  #{rc.rank} {rc.chunk.id} [{rc.chunk.lang}]{score} : {rc.chunk.text[:100].replace(chr(10), ' ')}…"
        )
    if ans.pipeline == "graphrag":
        ents = ans.extra.get("entities", [])
        rels = ans.extra.get("relations", [])
        print(
            f"  Graphe : {len(ents)} entités, {len(rels)} relations, mode={ans.extra.get('mode')}"
        )
        if ents:
            print("  Entités :", ", ".join(e["name"] for e in ents[:10]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--pipeline", default=None, help="rag | graphrag | both (défaut : config)")
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    s = load_settings(args.config)
    names = ["rag", "graphrag"] if args.pipeline == "both" else [args.pipeline or s.pipeline]
    for name in names:
        show(build_pipeline(name, s).answer(args.question))


if __name__ == "__main__":
    main()
