"""Tests du chargement des documents et du découpage, sans réseau."""

from __future__ import annotations

from pathlib import Path

from rag_uha.chunking import chunk_document, load_chunks
from rag_uha.loaders import load_document, load_documents
from rag_uha.tokens import get_encoding
from rag_uha.types import Document

FR = (
    "L'Université de Haute-Alsace est une université française située à Mulhouse et Colmar. "
    "Elle propose des formations en sciences, en droit, en économie et en lettres. "
) * 6
EN = (
    "The University of Upper Alsace is a French public university located in Mulhouse and Colmar. "
    "It offers programs in science, law, economics and humanities. "
) * 6


def test_load_txt_md_html(tmp_path: Path):
    (tmp_path / "doc-fr.txt").write_text(FR, encoding="utf-8")
    (tmp_path / "doc-en.md").write_text("# Programs\n\n" + EN, encoding="utf-8")
    (tmp_path / "page.html").write_text(
        "<html><head><title>Vie étudiante</title></head><body><nav>Menu Accueil Contact</nav>"
        f"<main><h1>Vie étudiante</h1><p>{FR}</p><p>{FR}</p></main>"
        "<footer>Mentions légales</footer></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    docs = load_documents(tmp_path)
    by_id = {d.doc_id: d for d in docs}
    assert set(by_id) == {"doc-fr", "doc-en", "page"}
    assert by_id["doc-fr"].lang == "fr" and by_id["doc-en"].lang == "en"
    assert by_id["doc-en"].title == "Programs"
    assert by_id["page"].fmt == "html" and "Haute-Alsace" in by_id["page"].text
    assert "Mentions légales" not in by_id["page"].text


def test_documents_csv_overrides(tmp_path: Path):
    (tmp_path / "doc-fr.txt").write_text(FR, encoding="utf-8")
    (tmp_path / "documents.csv").write_text(
        "doc_id,title,source,lang\ndoc-fr,Présentation UHA,https://www.uha.fr/x,en\n",
        encoding="utf-8",
    )
    docs = load_documents(tmp_path, tmp_path / "documents.csv")
    assert docs[0].title == "Présentation UHA"
    assert docs[0].source == "https://www.uha.fr/x"
    assert docs[0].lang == "en"  # forcé par le CSV


def test_empty_file_is_skipped(tmp_path: Path):
    (tmp_path / "vide.txt").write_text("court", encoding="utf-8")
    assert load_documents(tmp_path) == []


def test_pdf(tmp_path: Path):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    pdf = tmp_path / "blank.pdf"
    with pdf.open("wb") as f:
        w.write(f)
    doc = load_document(pdf)
    assert doc.fmt == "pdf" and doc.doc_id == "blank"


def test_chunking_sizes_and_overlap():
    enc = get_encoding()
    doc = Document("d", "t", FR * 20, "fr", "x", "txt")
    chunks = chunk_document(doc, chunk_tokens=100, overlap_tokens=20)
    assert [c.id for c in chunks][:3] == ["d::0", "d::1", "d::2"]
    assert all(len(enc.encode(c.text)) <= 103 for c in chunks)  # ±3 : re-tokenisation aux bords
    total = len(enc.encode(doc.text))
    assert len(chunks) == -(-max(total - 20, 1) // 80)  # ceil((n - overlap) / step)
    # le recouvrement : la fin du chunk 0 réapparaît au début du chunk 1
    assert chunks[0].text[-30:].split()[-2] in chunks[1].text[:200]


def test_short_doc_is_single_chunk():
    doc = Document("d", "t", "Petit texte.", "fr", "x", "txt")
    chunks = chunk_document(doc, chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) == 1 and chunks[0].text == "Petit texte."


def test_jsonl_roundtrip(tmp_path: Path):
    from rag_uha.chunking import build_chunks
    from rag_uha.config import load_settings

    s = load_settings()
    docs = [
        Document("a", "A", FR * 5, "fr", "x", "txt"),
        Document("b", "B", EN * 5, "en", "y", "md"),
    ]
    out = tmp_path / "chunks.jsonl"
    chunks = build_chunks(docs, out, s)
    assert load_chunks(out) == chunks
    assert {c.lang for c in chunks} == {"fr", "en"}


def test_technical_files_are_skipped(tmp_path: Path):
    (tmp_path / "doc-fr.txt").write_text(FR, encoding="utf-8")
    (tmp_path / "_scrape_report.md").write_text("# Rapport\n\n" + FR, encoding="utf-8")
    assert [d.doc_id for d in load_documents(tmp_path)] == ["doc-fr"]
