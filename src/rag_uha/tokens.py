"""Compteur de tokens commun (tiktoken `cl100k_base`).

tiktoken télécharge son vocabulaire au premier appel. LlamaIndex en embarque une copie :
on la déclare comme cache si rien n'est configuré, ce qui évite tout accès réseau
(utile hors ligne et derrière un proxy).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import tiktoken


@lru_cache(maxsize=4)
def get_encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    if "TIKTOKEN_CACHE_DIR" not in os.environ:
        try:
            import llama_index.core as lic

            cache = Path(lic.__file__).parent / "_static" / "tiktoken_cache"
            if cache.is_dir():
                os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
        except ImportError:
            pass
    return tiktoken.get_encoding(name)


def count_tokens(text: str, name: str = "cl100k_base") -> int:
    return len(get_encoding(name).encode(text))
