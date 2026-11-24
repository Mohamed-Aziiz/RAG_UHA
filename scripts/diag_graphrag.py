"""Diagnostic : ce que LightRAG fait de la question, etape par etape.

    uv run python scripts/diag_graphrag.py

Pour chaque question (avec et sans « uha ») et chaque mode (hybrid, mix), affiche :
1. les mots-cles extraits par le LLM ;
2. les entites trouvees dans le graphe ;
3. les relations retenues ;
4. les passages remontes, et si l'un d'eux parle du master data science.
Cout : quelques appels d'extraction de mots-cles (gpt-oss-20b), aucune generation de reponse.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from rag_uha.config import load_settings
from rag_uha.pipelines.graphrag import GraphRAGPipeline

QUESTIONS = [
    "est ce que uha propose une formation ingenieurie mathematique et data science",
    "est ce que vous propose une formation ingenieurie mathematique et data science",
]
MODES = ["hybrid", "mix"]
CIBLE = ("data science", "ingénierie mathématique", "ingenierie mathematique")


def parle_du_master(texte: str) -> bool:
    t = texte.lower()
    return any(c in t for c in CIBLE)


def main() -> None:
    settings = load_settings()
    gr = GraphRAGPipeline(settings)
    for q in QUESTIONS:
        for mode in MODES:
            print("=" * 100)
            print(f"QUESTION : {q}\nMODE     : {mode}")
            res = gr.retrieve_data(q, mode)
            data = res["data"]
            kw = res.get("metadata", {}).get("keywords", {})
            print(f"\n1. Mots-cles  haut niveau : {kw.get('high_level')}")
            print(f"              bas niveau  : {kw.get('low_level')}")

            ents = data.get("entities", [])
            print(f"\n2. Entites ({len(ents)}) - les 12 premieres :")
            if ents:
                print(f"   (champs disponibles : {sorted(ents[0].keys())})")
            for e in ents[:12]:
                extra = {k: e[k] for k in ("rank", "degree", "score") if k in e}
                print(f"   - {e.get('entity_name')} [{e.get('entity_type')}] {extra or ''}")

            rels = data.get("relationships", [])
            print(f"\n3. Relations ({len(rels)}) - les 10 premieres :")
            for r in rels[:10]:
                extra = {k: r[k] for k in ("rank", "degree", "weight", "score") if k in r}
                print(f"   - {r.get('src_id')} -> {r.get('tgt_id')} {extra or ''}")
            ents_cible = [
                e.get("entity_name")
                for e in ents
                if parle_du_master(f"{e.get('entity_name', '')} {e.get('description', '')}")
            ]
            print(f"   entites qui parlent du master : {ents_cible[:8] or 'AUCUNE'}")

            chunks = data.get("chunks", [])
            print(f"\n4. Passages ({len(chunks)}) :")
            for i, c in enumerate(chunks[:10], 1):
                cid = c.get("file_path") or c.get("chunk_id") or "?"
                flag = "  <== MASTER" if parle_du_master(c.get("content", "")) else ""
                print(f"   {i:>2}. {str(cid)[:80]}{flag}")
            rangs = [i for i, c in enumerate(chunks, 1) if parle_du_master(c.get("content", ""))]
            print(f"   passages qui parlent du master : rangs {rangs or 'AUCUN'}\n")


if __name__ == "__main__":
    main()
