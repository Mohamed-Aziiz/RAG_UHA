"""Étape 1 du pipeline : data/raw → data/chunks/chunks.jsonl (+ stats.json).

uv run python scripts/build_chunks.py
uv run python scripts/build_chunks.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

from rag_uha.chunking import build_chunks
from rag_uha.config import load_settings
from rag_uha.loaders import load_documents
from rag_uha.tokens import count_tokens

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    s = load_settings(args.config)

    docs = load_documents(s.paths.raw_dir, s.paths.documents_csv)
    if not docs:
        raise SystemExit(f"Aucun document dans {s.paths.raw_dir}")
    chunks = build_chunks(docs, s.paths.chunks_file, s)

    tokens_per_doc = {d.doc_id: count_tokens(d.text) for d in docs}
    stats = {
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "total_tokens": sum(tokens_per_doc.values()),
        "by_lang": dict(Counter(d.lang for d in docs)),
        "by_format": dict(Counter(d.fmt for d in docs)),
        "chunks_by_lang": dict(Counter(c.lang for c in chunks)),
        "chunking": s.chunking.model_dump(),
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "lang": d.lang,
                "fmt": d.fmt,
                "tokens": tokens_per_doc[d.doc_id],
                "chunks": sum(1 for c in chunks if c.doc_id == d.doc_id),
            }
            for d in docs
        ],
    }
    stats_file = s.paths.chunks_file.with_name("stats.json")
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"{stats['n_documents']} documents, {stats['total_tokens']} tokens, {stats['n_chunks']} chunks"
    )
    print(f"Langues : {stats['by_lang']} | formats : {stats['by_format']}")
    print(f"→ {s.paths.chunks_file}\n→ {stats_file}")
    for d in stats["documents"]:
        print(
            f"  {d['doc_id']:<40} {d['lang']}  {d['fmt']:<4} {d['tokens']:>6} tokens  {d['chunks']:>3} chunks"
        )


if __name__ == "__main__":
    main()
