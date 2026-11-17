"""Étape 3 : chunks.jsonl → graphe de connaissances LightRAG (extraction par LLM).

uv run python scripts/build_graph.py --limit 3     # test : 3 documents, mesure tokens / durée
uv run python scripts/build_graph.py               # indexation complète ; relançable (reprise)
uv run python scripts/build_graph.py --stats       # statistiques du graphe existant + doublons FR/EN
"""

from __future__ import annotations

import argparse
import json
import logging

from rag_uha.chunking import load_chunks
from rag_uha.config import load_settings
from rag_uha.indexing.graph_index import (
    build_graph,
    find_duplicate_entities,
    graph_stats,
    load_graph,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--limit", type=int, default=None, help="n'indexer que les N premiers documents")
    p.add_argument(
        "--stats", action="store_true", help="afficher les stats du graphe existant et sortir"
    )
    p.add_argument("--dedup-threshold", type=float, default=0.9)
    args = p.parse_args()
    s = load_settings(args.config)

    if args.stats:
        g = load_graph(s)
        st = graph_stats(g)
        print(json.dumps(st, ensure_ascii=False, indent=2))
        if g.number_of_nodes():
            from rag_uha.embeddings import get_embedder

            pairs = find_duplicate_entities(g, get_embedder(s), args.dedup_threshold)
            print(
                f"\n{len(pairs)} paires d'entités de même type avec similarité >= {args.dedup_threshold} :"
            )
            for a, b, sim in pairs[:40]:
                print(f"  {sim:.3f}  {a}  <->  {b}")
            out = s.paths.lightrag_dir / "duplicates.json"
            out.write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"→ {out}")
        return

    chunks = load_chunks(s.paths.chunks_file)
    r = build_graph(chunks, s, limit=args.limit)
    print("\n=== Rapport d'indexation ===")
    print(f"Chunks insérés : {r.n_chunks_requested} / {r.n_chunks_total}   modèle : {r.model}")
    print(f"Entités : {r.n_entities}   relations : {r.n_relations}")
    print(
        f"Tokens : {r.usage.prompt_tokens} entrée + {r.usage.completion_tokens} sortie "
        f"= {r.usage.total_tokens} ({r.usage.n_calls} appels)   durée : {r.duration_s:.0f}s"
    )
    if r.n_chunks_requested:
        per = r.usage.total_tokens / r.n_chunks_requested
        print(
            f"≈ {per:.0f} tokens / chunk  →  {r.n_chunks_total} chunks ≈ {per * r.n_chunks_total / 1000:.0f}k tokens"
        )
    print(f"→ {s.paths.lightrag_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
