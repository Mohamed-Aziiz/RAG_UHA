"""Recherche lexicale BM25, fusionnee avec la recherche vectorielle (RRF).

Pourquoi : un embedding resume un passage en un point de l'espace semantique. Sur une
enumeration ("... parcours Ingenierie mathematique et data sciences Mecanique • parcours
..."), ce resume ne ressemble a aucune question precise, et le passage ne remonte jamais,
quel que soit le nombre de candidats ou le reranker employe.

BM25 fait l'inverse : il compte les termes. "ingenierie mathematique" est un terme rare
dans le corpus, donc tres discriminant, et le passage qui le contient sort en tete.

Les deux sont complementaires : le vectoriel repond aux reformulations et au cross-lingue,
le lexical aux termes exacts et rares. On fusionne les deux classements par Reciprocal Rank
Fusion : score(d) = somme sur chaque classement de 1 / (k + rang(d)), k = 60 par defaut.
RRF ne compare pas des scores d'echelles differentes, seulement des rangs — c'est ce qui
le rend robuste ici, ou les scores cosinus tiennent dans une bande de 0,02.

Implementation BM25 Okapi maison (~40 lignes) : pas de dependance a installer.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from rag_uha.types import Chunk

_MOT = re.compile(r"[a-z0-9]+")


def tokenize(texte: str) -> list[str]:
    """Minuscules, accents retires, decoupe sur tout ce qui n'est pas alphanumerique."""
    sans_accent = "".join(
        c for c in unicodedata.normalize("NFD", texte.lower()) if unicodedata.category(c) != "Mn"
    )
    return _MOT.findall(sans_accent)


class BM25Index:
    """BM25 Okapi sur les chunks. Construction en memoire, instantanee pour ~1000 chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids = [c.id for c in chunks]
        self.docs = [tokenize(c.text) for c in chunks]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        n_docs = len(self.docs)
        df: Counter[str] = Counter()
        for f in self.freqs:
            df.update(f.keys())
        # IDF Okapi avec lissage : toujours positif, meme pour un terme present partout.
        self.idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, question: str, top_n: int) -> list[tuple[str, float]]:
        termes = [t for t in tokenize(question) if t in self.idf]
        if not termes:
            return []
        scores: list[tuple[str, float]] = []
        for i, freq in enumerate(self.freqs):
            s = 0.0
            for t in termes:
                f = freq.get(t)
                if not f:
                    continue
                norme = 1 - self.b + self.b * (self.lengths[i] / (self.avg_len or 1))
                s += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * norme)
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]


def rrf(classements: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion : fusionne plusieurs classements d'identifiants."""
    out: dict[str, float] = {}
    for cls in classements:
        for rang, cid in enumerate(cls, start=1):
            out[cid] = out.get(cid, 0.0) + 1.0 / (k + rang)
    return out
