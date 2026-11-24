"""Diagnostic 3 : pourquoi le passage du master disparait meme quand l'entite est trouvee.

    uv run python scripts/diag_passages.py

Le diagnostic 2 a montre que, en donnant les bons mots-cles, l'entite
« parcours Ingenierie mathematique et data sciences » est bien trouvee, mais son passage n'arrive
pas dans le contexte. Cause probable : LightRAG choisit les passages lies aux entites par
similarite avec la QUESTION BRUTE (kg_chunk_pick_method = VECTOR), et la question brute contient
« uha ». On compare ici avec la methode WEIGHT (passages choisis selon l'ordre des entites
trouvees, sans la question brute). Aucun appel LLM. Resultat dans diag_passages.txt (UTF-8).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_uha.config import load_settings
from rag_uha.pipelines.graphrag import GraphRAGPipeline, _run

AVEC = "est ce que uha propose une formation ingenieurie mathematique et data science"
SANS = "est ce que vous propose une formation ingenieurie mathematique et data science"
HL = ["education", "engineering mathematics", "data science training"]
PRECIS = ["formation", "ingénierie mathématique", "data science"]
# (question, mots-cles imposes ou None = extraction du LLM, methode de choix des passages)
CAS = [
    ("1. avec uha | UHA en haut niveau | VECTOR", AVEC, (["UHA"] + HL, PRECIS), "VECTOR"),
    ("2. avec uha | UHA en haut niveau | WEIGHT", AVEC, (["UHA"] + HL, PRECIS), "WEIGHT"),
    ("3. avec uha | UHA en bas niveau + termes precis | WEIGHT", AVEC, (HL, ["UHA"] + PRECIS), "WEIGHT"),
    ("4. avec uha | mots-cles du LLM (actuel) | WEIGHT", AVEC, None, "WEIGHT"),
    ("5. sans uha | mots-cles du LLM | VECTOR (reference : marchait)", SANS, None, "VECTOR"),
    ("6. sans uha | mots-cles du LLM | WEIGHT (ne doit pas regresser)", SANS, None, "WEIGHT"),
]
CIBLE = ("data science", "ingénierie mathématique", "ingenierie mathematique")


def parle_du_master(t: str) -> bool:
    t = t.lower()
    return any(c in t for c in CIBLE)


def main() -> None:
    gr = GraphRAGPipeline(load_settings())
    config = gr.rag.text_chunks.global_config  # dictionnaire partage par tous les stockages
    lignes = []
    for nom, q, kw, methode in CAS:
        config["kg_chunk_pick_method"] = methode
        param = gr._param("hybrid")
        if kw:
            param.hl_keywords, param.ll_keywords = kw
        res = _run(gr.rag.aquery_data(q, param=param))
        data = res.get("data", {})
        got = res.get("metadata", {}).get("keywords", {})
        ents, rels, chunks = data.get("entities", []), data.get("relationships", []), data.get("chunks", [])
        rangs = [i for i, c in enumerate(chunks, 1) if parle_du_master(c.get("content", ""))]
        ent_master = [i for i, e in enumerate(ents, 1) if parle_du_master(e.get("entity_name", ""))]
        lignes += [
            "=" * 90,
            nom,
            f"  question : {q}",
            f"  mots-cles : haut {got.get('high_level')} | bas {got.get('low_level')}",
            f"  entites {len(ents)} (master aux rangs {ent_master or 'AUCUN'}) | relations {len(rels)}",
            f"  passages {len(chunks)} ; passages du master aux rangs {rangs or 'AUCUN'}",
        ]
    out = ROOT / "diag_passages.txt"
    out.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"Ecrit : {out}")


if __name__ == "__main__":
    main()
