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
import fnmatch
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
    """Texte d'un PDF dans l'ordre VISUEL de lecture (PyMuPDF, blocs tries par position).

    pypdf.extract_text() suit l'ordre interne du fichier, pas l'ordre de lecture. Sur la
    plaquette de l'offre de formation, les titres de section (LICENCES PROFESSIONNELLES,
    MASTERS), dessines a part, ressortaient ensemble en tete de page, detaches de leurs
    listes : le texte indexe ne disait plus qu'une formation etait un master, et le LLM
    refusait a juste titre d'affirmer ce que son contexte ne disait pas.
    Le mode `layout` de pypdf remet l'ordre mais coupe les mots ("infor matique",
    "p arcour s"), ce qui casse BM25 et les embeddings. PyMuPDF fait les deux correctement.
    """
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        pages = [page.get_text("text", sort=True).strip() for page in doc]
        raw_title = (doc.metadata or {}).get("title") or ""
    title = raw_title.strip() or None
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


def load_documents(
    raw_dir: Path, documents_csv: Path | None = None, exclude: list[str] | None = None
) -> list[Document]:
    """Tous les documents du dossier, triés par nom. Les fichiers vides sont ignorés avec un avertissement."""
    overrides = read_documents_csv(documents_csv) if documents_csv else {}
    docs: list[Document] = []
    n_excluded = 0
    for path in sorted(raw_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED
            or path.name == "documents.csv"
            or path.name.startswith(("_", "."))  # fichiers techniques (_scrape_report.md…)
        ):
            continue
        if exclude and any(fnmatch.fnmatch(path.stem, pat) for pat in exclude):
            n_excluded += 1
            continue
        doc = load_document(path, overrides.get(path.stem))
        if len(doc.text) < 200:
            log.warning(
                "Document presque vide ignoré : %s (%d caractères)", path.name, len(doc.text)
            )
            continue
        docs.append(doc)
    if n_excluded:
        log.info("%d document(s) exclu(s) par corpus.exclude", n_excluded)
    if len({d.doc_id for d in docs}) != len(docs):
        raise ValueError(
            "Deux fichiers ont le même nom sans extension : les doc_id doivent être uniques."
        )
    return docs
