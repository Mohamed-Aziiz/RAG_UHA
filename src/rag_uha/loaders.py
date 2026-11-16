"""Chargement des documents scrapés : .txt, .md, .html, .pdf → `Document`.

Entrée : le dossier `paths.raw_dir`. Chaque fichier devient un document dont `doc_id` est le
nom du fichier sans extension (ex. `master-data-science.html` → `master-data-science`).

Un fichier facultatif `documents.csv` (colonnes : doc_id, title, source, lang) permet de
forcer le titre, l'URL d'origine et la langue d'un document. Sans lui, le titre est déduit
du contenu et la langue détectée automatiquement (langdetect).

Usage :
    docs = load_documents(settings.paths.raw_dir, settings.paths.documents_csv)
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from rag_uha.types import DocFormat, Document, Lang

log = logging.getLogger(__name__)

SUPPORTED: dict[str, DocFormat] = {
    ".txt": "txt",
    ".md": "md",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


# ------------------------------------------------------------------ extraction par format
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_html(path: Path) -> tuple[str, str | None]:
    """Texte principal de la page (menus, pieds de page et scripts retirés) et titre."""
    import trafilatura

    html = _read_text(path)
    text = trafilatura.extract(
        html, include_comments=False, include_tables=True, include_links=False, favor_recall=True
    )
    if not text:  # page sans contenu « article » reconnu : on prend tout le texte visible
        text = trafilatura.html2txt(html)
    meta = trafilatura.extract_metadata(html)
    title = meta.title if meta and meta.title else None
    return text or "", title


def _read_pdf(path: Path) -> tuple[str, str | None]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or None
    return "\n\n".join(p for p in pages if p), title


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if 3 <= len(line) <= 120:
            return line
    return fallback


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_lang(text: str) -> Lang:
    """fr ou en. En cas de doute (texte trop court, autre langue), fr."""
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0  # résultat reproductible
    try:
        code = detect(text[:5000])
    except Exception:  # noqa: BLE001 — texte vide ou illisible
        return "fr"
    return "en" if code == "en" else "fr"


# ------------------------------------------------------------------ API
def load_document(path: Path, overrides: dict[str, str] | None = None) -> Document:
    fmt = SUPPORTED.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"Format non géré : {path.name}")

    title: str | None = None
    if fmt == "html":
        text, title = _read_html(path)
    elif fmt == "pdf":
        text, title = _read_pdf(path)
    else:
        text = _read_text(path)
    text = _clean(text)

    o = overrides or {}
    doc_id = path.stem
    return Document(
        doc_id=doc_id,
        title=o.get("title") or title or _title_from_text(text, doc_id),
        text=text,
        lang=o.get("lang") or detect_lang(text),  # type: ignore[arg-type]
        source=o.get("source") or str(path),
        fmt=fmt,
    )


def read_documents_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        r["doc_id"].strip(): {k: (v or "").strip() for k, v in r.items()}
        for r in rows
        if r.get("doc_id")
    }


def load_documents(raw_dir: Path, documents_csv: Path | None = None) -> list[Document]:
    """Tous les documents du dossier, triés par nom. Les fichiers vides sont ignorés avec un avertissement."""
    overrides = read_documents_csv(documents_csv) if documents_csv else {}
    docs: list[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED
            or path.name == "documents.csv"
        ):
            continue
        doc = load_document(path, overrides.get(path.stem))
        if len(doc.text) < 200:
            log.warning(
                "Document presque vide ignoré : %s (%d caractères)", path.name, len(doc.text)
            )
            continue
        docs.append(doc)
    if len({d.doc_id for d in docs}) != len(docs):
        raise ValueError(
            "Deux fichiers ont le même nom sans extension : les doc_id doivent être uniques."
        )
    return docs
