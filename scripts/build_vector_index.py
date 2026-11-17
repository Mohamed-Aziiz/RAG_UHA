"""Étape 2 : chunks.jsonl → index ChromaDB (RAG classique). 0 token de LLM.

uv run python scripts/build_vector_index.py
"""

from __future__ import annotations

import argparse
import logging

from rag_uha.chunking import load_chunks
from rag_uha.config import load_settings
from rag_uha.indexing.vector_index import build_vector_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    s = load_settings(args.config)
    chunks = load_chunks(s.paths.chunks_file)
    n = build_vector_index(chunks, s)
    print(f"{n} chunks indexés dans {s.paths.chroma_dir} (collection {s.rag.collection})")


if __name__ == "__main__":
    main()
