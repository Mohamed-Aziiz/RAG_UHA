"""Enregistre les réponses des deux pipelines à quelques questions → data/demo_cache.json
(secours pour la démo si l'API tombe).

    uv run python scripts/record_demo.py "Question 1" "Question 2" …
    uv run python scripts/record_demo.py --from-file eval/demo_questions.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_uha.config import load_settings
from rag_uha.pipelines.base import build_pipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("questions", nargs="*")
    p.add_argument("--from-file", default=None)
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    s = load_settings(args.config)
    qs = list(args.questions)
    if args.from_file:
        qs += [
            line.strip()
            for line in Path(args.from_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not qs:
        raise SystemExit("Aucune question.")

    out = s.paths.demo_cache_file
    cache = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for name in ("rag", "graphrag"):
        pipe = build_pipeline(name, s)
        cache.setdefault(name, {})
        for q in qs:
            if q in cache[name]:
                continue
            print(f"[{name}] {q}")
            cache[name][q] = pipe.answer(q).to_dict()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out} ({sum(len(v) for v in cache.values())} réponses)")


if __name__ == "__main__":
    main()
