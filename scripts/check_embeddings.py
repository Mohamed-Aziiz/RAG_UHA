"""Vérifie l'API d'embeddings : dimension, multilinguisme, absence de troncature.

    uv run python scripts/check_embeddings.py

Trois contrôles :
1. dimension 1024, vecteurs normalisés ;
2. multilingue : « Université de Haute-Alsace » proche de « University of Upper Alsace » et
   loin d'un texte sans rapport ;
3. troncature : un texte de ~1 200 tokens dont la FIN seule parle d'un sujet précis doit être
   plus proche d'une requête sur ce sujet que ses 400 premiers tokens seuls. Si ce n'est pas le
   cas, l'API coupe les textes longs : réduire `chunking.chunk_tokens`.
"""

from __future__ import annotations

import numpy as np

from rag_uha.config import load_settings
from rag_uha.embeddings import get_embedder


def main() -> None:
    s = load_settings()
    emb = get_embedder(s)

    fr = "L'Université de Haute-Alsace propose des formations à Mulhouse et Colmar."
    en = "The University of Upper Alsace offers degree programs in Mulhouse and Colmar."
    other = "La recette de la tarte flambée demande de la crème, des oignons et des lardons."
    v = emb.embed_documents([fr, en, other])
    print(f"1. dimension : {v.shape[1]}  | normes : {np.linalg.norm(v, axis=1).round(3)}")
    print(f"2. sim(FR, EN) = {float(v[0] @ v[1]):.3f}   sim(FR, autre) = {float(v[0] @ v[2]):.3f}")

    filler = (
        "Les services administratifs traitent les dossiers d'inscription et les demandes de "
        "bourse. Les étudiants peuvent consulter leur emploi du temps en ligne. "
    )
    tail = (
        "Le laboratoire de photochimie étudie les polymères sensibles à la lumière ultraviolette "
        "pour l'impression 3D et les revêtements industriels. "
    )
    body = filler * 40  # ~1 000 tokens
    long_text = body + tail * 4
    n_long = emb.count_tokens(long_text)
    short_text = long_text[: len(long_text) // 3]
    q = emb.embed_query("recherche sur les polymères photosensibles et l'impression 3D")
    vl, vs = emb.embed_documents([long_text, short_text])
    sim_long, sim_short = float(q @ vl), float(q @ vs)
    print(
        f"3. texte de {n_long} tokens : sim(requête, long) = {sim_long:.3f} ; "
        f"sim(requête, début seul) = {sim_short:.3f}"
    )
    if sim_long > sim_short + 0.02:
        print("   OK : la fin du texte long est bien prise en compte (pas de troncature).")
    else:
        print("   ATTENTION : troncature probable. Réduire chunking.chunk_tokens (ex. 500).")

    print(f"\nTextes embarqués : {emb.n_texts}, tokens estimés : {emb.n_tokens_est}")


if __name__ == "__main__":
    main()
