"""Appelle réellement les trois rôles LLM et affiche tokens et latence.

uv run python scripts/check_llm.py
uv run python scripts/check_llm.py --roles query --prompt "Qu'est-ce que l'UHA ?"
"""

from __future__ import annotations

import argparse
import logging
import time

from rag_uha.config import load_settings
from rag_uha.llm import get_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--roles", nargs="+", default=["indexing", "query", "judge"])
    p.add_argument(
        "--prompt", default="Réponds en une phrase : qu'est-ce qu'un graphe de connaissances ?"
    )
    args = p.parse_args()

    settings = load_settings(args.config)
    for role in args.roles:
        llm = get_llm(role, settings)
        t0 = time.perf_counter()
        text = llm.complete(args.prompt, system="Tu es concis.")
        dt = time.perf_counter() - t0
        u = llm.last_usage
        print(
            f"\n[{role}] {llm.cfg.provider} / {llm.model}  —  {dt:.1f}s, "
            f"{u.prompt_tokens} + {u.completion_tokens} tokens"
        )
        print("   ", text[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
