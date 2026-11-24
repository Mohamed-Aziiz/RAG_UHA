"""Diagnostic 2 : la meme question avec « UHA », en imposant les mots-cles a LightRAG.

    uv run python scripts/diag_motscles.py

Le diagnostic 1 a montre que, avec « uha » dans la question, le LLM met SEULEMENT « UHA » dans les
mots-cles bas niveau (ceux qui servent a chercher les entites). Les termes precis (ingenierie
mathematique, data science) ne servent alors pas a chercher les entites. Ici on fournit nous-memes
les mots-cles (QueryParam.hl_keywords / ll_keywords, LightRAG saute alors l'extraction) pour voir
quelle repartition retrouve le master, en GARDANT « UHA ».
Aucun appel LLM : seulement des embeddings. Resultat ecrit dans diag_motscles.txt (UTF-8).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_uha.config import load_settings
from rag_uha.pipelines.graphrag import GraphRAGPipeline, _run

Q = "est ce que uha propose une formation ingenieurie mathematique et data science"
HL = ["education", "engineering mathematics", "data science training"]
PRECIS = ["formation", "ingénierie mathématique", "data science"]
VARIANTES = {
    "A. extraction du LLM (actuel)": None,
    "B. bas niveau = UHA + termes precis": (HL, ["UHA"] + PRECIS),
    "C. bas niveau = termes precis + UHA (UHA en dernier)": (HL, PRECIS + ["UHA"]),
    "D. UHA en haut niveau, termes precis en bas niveau": (["UHA"] + HL, PRECIS),
    "E. termes precis seuls (reference, sans UHA)": (HL, PRECIS),
}
CIBLE = ("data science", "ingénierie mathématique", "ingenierie mathematique")


def parle_du_master(t: str) -> bool:
    t = t.lower()
    return any(c in t for c in CIBLE)


def main() -> None:
    settings = load_settings()
    gr = GraphRAGPipeline(settings)
    lignes = [f"QUESTION : {Q}\n"]
    for nom, kw in VARIANTES.items():
        param = gr._param("hybrid")
        if kw:
            param.hl_keywords, param.ll_keywords = kw
        res = _run(gr.rag.aquery_data(Q, param=param))
        data = res.get("data", {})
        got = res.get("metadata", {}).get("keywords", {})
        ents, rels, chunks = (
            data.get("entities", []),
            data.get("relationships", []),
            data.get("chunks", []),
        )
        rangs = [i for i, c in enumerate(chunks, 1) if parle_du_master(c.get("content", ""))]
        ent_master = [e["entity_name"] for e in ents if parle_du_master(e.get("entity_name", ""))]
        hub = sum(1 for r in rels if "UHA" in f"{r.get('src_id')}{r.get('tgt_id')}")
        lignes += [
            "=" * 90,
            nom,
            f"  mots-cles utilises : haut {got.get('high_level')} | bas {got.get('low_level')}",
            f"  entites : {len(ents)} (qui nomment le master : {ent_master or 'AUCUNE'})",
            f"  relations : {len(rels)} (dont {hub} touchent un noeud « UHA »)",
            f"  passages : {len(chunks)} ; passages du master aux rangs {rangs or 'AUCUN'}",
            "  premiers passages : "
            + ", ".join(str(c.get("file_path", "?"))[:45] for c in chunks[:4]),
        ]
    out = ROOT / "diag_motscles.txt"
    out.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"Ecrit : {out}")


if __name__ == "__main__":
    main()
