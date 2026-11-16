"""Découpage commun en chunks, utilisé par le RAG ET le GraphRAG.

Fenêtre glissante de `chunk_tokens` tokens avec `overlap_tokens` de recouvrement, comptés avec
tiktoken (`cl100k_base`). C'est exactement la méthode par défaut de LightRAG : on la
reproduit ici pour que les deux architectures voient les mêmes morceaux de texte.

Chaque chunk a un identifiant stable `<doc_id>::<position>` qui sert de source dans les deux
pipelines (métadonnée ChromaDB côté RAG, `file_path` côté LightRAG).

Usage :
    chunks = build_chunks(docs, settings.paths.chunks_file, settings)
    chunks = load_chunks(settings.paths.chunks_file)
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_uha.config import Settings
from rag_uha.tokens import get_encoding
from rag_uha.types import Chunk, Document


def chunk_document(
    doc: Document,
    chunk_tokens: int = 1200,
    overlap_tokens: int = 100,
    tokenizer: str = "cl100k_base",
) -> list[Chunk]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens doit être inférieur à chunk_tokens")
    enc = get_encoding(tokenizer)
    tokens = enc.encode(doc.text)
    step = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    for position, start in enumerate(range(0, max(len(tokens), 1), step)):
        piece = tokens[start : start + chunk_tokens]
        text = enc.decode(piece).strip()
        if text:
            chunks.append(
                Chunk(
                    id=f"{doc.doc_id}::{position}",
                    doc_id=doc.doc_id,
                    text=text,
                    lang=doc.lang,
                    position=position,
                )
            )
        if start + chunk_tokens >= len(tokens):
            break
    return chunks


def build_chunks(docs: list[Document], out_file: Path, settings: Settings) -> list[Chunk]:
    """Découpe tous les documents et écrit un JSONL (une ligne par chunk)."""
    c = settings.chunking
    chunks = [
        ch
        for doc in docs
        for ch in chunk_document(doc, c.chunk_tokens, c.overlap_tokens, c.tokenizer)
    ]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
    return chunks


def load_chunks(chunks_file: Path) -> list[Chunk]:
    with chunks_file.open(encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
