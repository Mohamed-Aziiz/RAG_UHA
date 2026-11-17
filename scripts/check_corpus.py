"""
check_corpus.py - Controle qualite de data/raw/ avant le chunking.

A lancer apres scrape_uha.py, avant build_chunks.py. Verifie que le corpus est
chargeable, coherent avec documents.csv, et conforme aux cibles du plan.
Sort en code 1 si une anomalie bloquante est trouvee.

    uv run python scripts/check_corpus.py
    uv run python scripts/check_corpus.py --raw-dir data/raw --show 5
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

REQUIRED_KEYS = ("doc_id", "title", "url", "lang", "fmt", "category")
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def count_tokens(text: str) -> int:
    if _ENC is not None:
        try:
            return len(_ENC.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Lecture minimale du frontmatter : cles simples, valeurs scalaires."""
    m = FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        meta[key.strip()] = value
    return meta, text[m.end() :]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Controle qualite du corpus brut.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument(
        "--show",
        type=int,
        default=3,
        help="nombre de documents dont on affiche un extrait",
    )
    ap.add_argument("--min-docs", type=int, default=30, help="cible du plan")
    ap.add_argument("--min-tokens", type=int, default=80_000, help="cible du plan")
    ap.add_argument("--min-en-ratio", type=float, default=1 / 3, help="cible du plan")
    args = ap.parse_args(argv)

    raw = Path(args.raw_dir)
    csv_path = raw / "documents.csv"
    errors: list[str] = []
    warnings: list[str] = []

    if not csv_path.exists():
        print(f"ERREUR : {csv_path} introuvable. Lance d'abord scrape_uha.py.")
        return 1

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("doc_id")]

    print(f"Index : {len(rows)} documents declares dans {csv_path}\n")

    seen_ids: set[str] = set()
    total_tokens = 0
    by_lang: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    previews: list[tuple[str, str]] = []

    for row in rows:
        doc_id, fmt = row["doc_id"], row.get("fmt", "")
        if doc_id in seen_ids:
            errors.append(f"{doc_id} : doc_id en double dans documents.csv")
        seen_ids.add(doc_id)

        path = raw / f"{doc_id}.{'md' if fmt == 'md' else fmt}"
        if not path.exists():
            errors.append(f"{doc_id} : fichier {path.name} absent")
            continue

        if fmt == "md":
            text = path.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(text)
            if not meta:
                errors.append(f"{doc_id} : frontmatter YAML absent")
                continue
            for key in REQUIRED_KEYS:
                if not meta.get(key):
                    errors.append(f"{doc_id} : cle '{key}' absente du frontmatter")
            if meta.get("doc_id") != doc_id:
                errors.append(f"{doc_id} : doc_id du frontmatter != nom de fichier")
            if meta.get("lang") not in ("fr", "en"):
                errors.append(f"{doc_id} : langue '{meta.get('lang')}' hors (fr, en)")
            if len(body.strip()) < 200:
                warnings.append(f"{doc_id} : corps tres court ({len(body.strip())} caracteres)")
            if re.search(
                r"(?im)^\s*(cookies?|accepter tous|menu principal|aller au contenu)\b",
                body,
            ):
                warnings.append(f"{doc_id} : reste de navigation ou banniere cookies dans le texte")
            tokens = count_tokens(body)
            previews.append((doc_id, body.strip()[:400]))

        elif fmt == "pdf":
            if PdfReader is None:
                warnings.append(f"{doc_id} : pypdf absent, PDF non verifie")
                continue
            try:
                reader = PdfReader(str(path))
                text = "\n".join((p.extract_text() or "") for p in reader.pages[:80])
            except Exception as exc:
                errors.append(f"{doc_id} : PDF illisible ({exc})")
                continue
            n_pages = len(reader.pages)
            if len(text.strip()) < 200:
                errors.append(
                    f"{doc_id} : PDF sans couche texte sur {n_pages} pages "
                    f"(scan) -> a ecarter du corpus ou a passer par OCR"
                )
                continue
            if len(text) / max(1, n_pages) < 150:
                warnings.append(f"{doc_id} : tres peu de texte par page ({n_pages} pages)")
            tokens = count_tokens(text)
            previews.append((doc_id, text.strip()[:400]))
        else:
            warnings.append(f"{doc_id} : format '{fmt}' non gere par loaders.py")
            continue

        total_tokens += tokens
        declared = int(row.get("n_tokens_est") or 0)
        if declared and abs(declared - tokens) > max(50, 0.25 * declared):
            warnings.append(
                f"{doc_id} : tokens recomptes {tokens} vs {declared} declares dans le CSV"
            )
        by_lang[row.get("lang", "?")] = by_lang.get(row.get("lang", "?"), 0) + 1
        by_cat[row.get("category", "?")] = by_cat.get(row.get("category", "?"), 0) + 1

    # Fichiers presents sur le disque mais absents de l'index
    on_disk = {p.stem for p in raw.glob("*.md") if not p.name.startswith("_")}
    on_disk |= {p.stem for p in raw.glob("*.pdf")}
    for orphan in sorted(on_disk - seen_ids):
        warnings.append(f"{orphan} : fichier present mais absent de documents.csv")

    # --- Bilan ----------------------------------------------------------
    n_docs = len(seen_ids)
    n_en = by_lang.get("en", 0)
    en_ratio = n_en / n_docs if n_docs else 0.0
    chunks = total_tokens // 1100

    print("Corpus")
    print(f"  documents        : {n_docs}")
    print(f"  tokens (recomptes): {total_tokens}")
    print(f"  chunks a 1200/100 : ~{chunks}")
    print(f"  langues          : {by_lang}")
    print(f"  categories       : {by_cat}")
    print()

    print("Cibles du plan")

    def verdict(ok: bool) -> str:
        return "OK  " if ok else "SOUS LA CIBLE"

    print(f"  {verdict(n_docs >= args.min_docs)} documents {n_docs} / {args.min_docs}")
    print(f"  {verdict(total_tokens >= args.min_tokens)} tokens {total_tokens} / {args.min_tokens}")
    print(
        f"  {verdict(en_ratio >= args.min_en_ratio)} part anglaise "
        f"{en_ratio:.0%} / {args.min_en_ratio:.0%}"
    )
    print(f"  {verdict(len(by_cat) >= 3)} themes {len(by_cat)} / 3")
    print()

    if args.show:
        print(
            f"Extraits ({min(args.show, len(previews))} documents) "
            "- relis-les, c'est le controle visuel du plan"
        )
        print("-" * 70)
        step = max(1, len(previews) // max(1, args.show))
        for doc_id, preview in previews[::step][: args.show]:
            print(f"[{doc_id}]")
            print(preview.replace("\n", " ")[:360])
            print("-" * 70)
        print()

    if warnings:
        print(f"Avertissements ({len(warnings)}) :")
        for w in warnings[:30]:
            print(f"  - {w}")
        if len(warnings) > 30:
            print(f"  ... et {len(warnings) - 30} autres")
        print()

    if errors:
        print(f"ERREURS BLOQUANTES ({len(errors)}) :")
        for e in errors[:30]:
            print(f"  - {e}")
        print("\nCorrige-les avant build_chunks.py.")
        return 1

    print("Aucune erreur bloquante. Le corpus est pret pour build_chunks.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
