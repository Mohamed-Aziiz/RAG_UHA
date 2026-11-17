"""
scrape_uha.py - Collecte du corpus documentaire UHA pour le projet RAG / GraphRAG.

Tache T1 du plan : parcourt une section du site de l'UHA, extrait le texte propre
de chaque page, telecharge les PDF rencontres, et ecrit :

    data/raw/<doc_id>.md      une page web -> Markdown + frontmatter YAML
    data/raw/<doc_id>.pdf     les PDF, tels quels (pypdf les lira plus tard)
    data/raw/documents.csv    l'index du corpus (doc_id, titre, url, langue, format...)
    data/raw/_scrape_state.json   etat du crawl, permet la reprise
    data/raw/_scrape_report.md    rapport lisible de la collecte

Pourquoi le Markdown : il garde les titres (h1/h2/h3), les listes et les tableaux.
Le chunking de 1200 tokens coupe alors sur des frontieres de sens et non au
milieu d'une phrase, et les entites extraites par LightRAG heritent du contexte
de leur section. Le frontmatter YAML transporte les metadonnees (url, langue,
categorie) jusqu'a loaders.py sans fichier annexe.

Dependances : httpx, lxml, trafilatura, langdetect, tiktoken, pypdf.
Toutes figurent deja dans pyproject.toml. trafilatura, langdetect, tiktoken et
pypdf sont facultatives : le script se rabat sur des solutions de secours.

Exemples (PowerShell) :

    # 1. Reconnaissance : rien n'est ecrit, on voit ce qui serait collecte
    uv run python scripts/scrape_uha.py --dry-run

    # 2. Collecte reelle de la section formations
    uv run python scripts/scrape_uha.py --category formations --max-pages 40

    # 3. Section anglaise, ajoutee au meme corpus (le CSV est fusionne)
    uv run python scripts/scrape_uha.py `
        --start-url https://www.uha.fr/en/study-1.html `
        --category formations --lang-hint en --max-pages 20

    # 4. Reprise apres interruption (rien n'est retelecharge)
    uv run python scripts/scrape_uha.py --category formations --max-pages 40
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import ssl
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import httpx

# --------------------------------------------------------------------------
# Dependances facultatives
# --------------------------------------------------------------------------

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from lxml import html as lxml_html
except ImportError:
    lxml_html = None

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
except ImportError:
    detect = None

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# --------------------------------------------------------------------------
# Valeurs par defaut
# --------------------------------------------------------------------------

DEFAULT_START = "https://www.uha.fr/fr/formation-1.html"

# Un lien n'est suivi comme page que s'il correspond a l'un de ces motifs.
# Resserre le crawl sur l'offre de formation au lieu d'aspirer tout le site.
DEFAULT_ALLOW = [
    r"/fr/formation",
    r"/fr/les-formations",
    r"/fr/diplome",
    r"/fr/licence",
    r"/fr/master",
    r"/fr/but\b",
    r"/fr/doctorat",
    r"/fr/candidat",
    r"/fr/inscription",
    r"/en/stud",
    r"/en/educat",
    r"/en/programme",
    r"/en/degree",
    r"/en/admission",
]

# Jamais suivi, meme si le motif precedent correspond.
DEFAULT_DENY = [
    r"\.(?:jpe?g|png|gif|svg|webp|ico|css|js|zip|rar|7z|mp[34]|avi|mov|docx?|xlsx?|pptx?|odt|ods)(?:$|\?)",
    r"/(?:fileadmin|typo3|typo3conf|typo3temp)/.*\.(?!pdf)",
    r"[?&]tx_[a-z_]+\[",  # formulaires et paginations TYPO3
    r"[?&](?:print|pdf_view|cHash)=",
    r"/(?:login|logout|connexion|recherche|search|mentions-legales|plan-du-site)",
    r"/(?:actualite|agenda|evenement|news)",
    r"#",
]

# Hotes autorises pour le telechargement des PDF (sous-domaines de l'UHA inclus).
DEFAULT_PDF_HOSTS = r"(?:^|\.)uha\.fr$"

USER_AGENT = (
    "UHA-RAG-student-project/1.0 "
    "(projet universitaire M1 Data Science; contact: voir README du depot)"
)

TEXT_HTML = ("text/html", "application/xhtml+xml")


def build_verify(args: argparse.Namespace) -> tuple[object, str]:
    """Choisit le magasin de certificats et renvoie (valeur pour httpx, libelle).

    httpx valide par defaut avec certifi, un paquet fige de racines publiques.
    Un antivirus, un pare-feu d'entreprise ou un reseau filtrant remplace le
    certificat du site par le sien : sa racine est dans le magasin de Windows,
    jamais dans certifi, d'ou un echec de Python la ou le navigateur passe.
    truststore delegue la validation au systeme, exactement comme le navigateur.
    """
    if args.ca_bundle:
        path = Path(args.ca_bundle)
        if not path.exists():
            print(f"[tls] paquet de certificats introuvable : {path}")
            sys.exit(2)
        return ssl.create_default_context(cafile=str(path)), f"paquet fourni ({path.name})"

    if args.ssl == "certifi":
        return True, "certifi (defaut de httpx)"

    try:
        import truststore

        return (
            truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            "magasin de certificats du systeme (truststore)",
        )
    except ImportError:
        if args.ssl == "system":
            return (
                ssl.create_default_context(),
                "magasin de certificats du systeme (module ssl)",
            )
    return True, "certifi (defaut de httpx)"


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------


@dataclass
class DocRecord:
    """Une ligne de documents.csv."""

    doc_id: str
    title: str
    url: str
    lang: str
    fmt: str  # "md" (page web) ou "pdf"
    category: str
    n_chars: int
    n_tokens_est: int
    n_pages: int  # 0 pour les pages web
    sha256: str
    scraped_at: str


@dataclass
class CrawlStats:
    pages_ok: int = 0
    pages_skipped_thin: int = 0
    pages_failed: int = 0
    pdfs_ok: int = 0
    pdfs_failed: int = 0
    pdfs_too_large: int = 0
    blocked_by_robots: int = 0
    already_present: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Utilitaires URL
# --------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Supprime l'ancre, les slashs finaux superflus, normalise le schema/hote."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    netloc = netloc.removesuffix(":443")
    netloc = netloc.removesuffix(":80")
    path = re.sub(r"/{2,}", "/", parsed.path) or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def matches_any(url: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(url) for p in patterns)


def slugify(value: str, max_len: int = 72) -> str:
    """Transforme un texte en identifiant sur et stable (compatible Windows)."""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    value = re.sub(r"-{2,}", "-", value)
    return value[:max_len].strip("-") or "doc"


def url_to_doc_id(url: str, taken: set[str]) -> str:
    """doc_id stable derive de l'URL, unique dans le corpus."""
    parsed = urlparse(url)
    path = parsed.path
    is_pdf = path.lower().endswith(".pdf")
    for ext in (".html", ".htm", ".php", ".pdf"):
        if path.lower().endswith(ext):
            path = path[: -len(ext)]
            break
    # Un PDF garde son seul nom de fichier : "/fileadmin/x/reglement.pdf" -> "reglement".
    base = slugify(path.rsplit("/", 1)[-1] if is_pdf else path)
    if parsed.query:
        base = f"{base}-{hashlib.sha1(parsed.query.encode()).hexdigest()[:6]}"
    if base in taken:
        base = f"{base}-{hashlib.sha1(url.encode()).hexdigest()[:6]}"
    taken.add(base)
    return base


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    if _ENC is not None:
        try:
            return len(_ENC.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


def detect_lang(text: str, url: str, html_lang: str | None, hint: str | None) -> str:
    """Priorite : attribut lang du HTML > segment d'URL > langdetect > indice CLI."""
    if html_lang:
        code = html_lang.strip().lower()[:2]
        if code in ("fr", "en"):
            return code
    path = urlparse(url).path.lower()
    if re.search(r"^/en(?:/|$)|/english/", path):
        return "en"
    if re.search(r"^/fr(?:/|$)", path):
        return "fr"
    sample = text.strip()[:3000]
    if detect is not None and len(sample) > 80:
        try:
            code = detect(sample)[:2]
            if code in ("fr", "en"):
                return code
        except Exception:
            pass
    return (hint or "fr")[:2]


def extract_html_lang(html_text: str) -> str | None:
    m = re.search(r"<html[^>]*\blang\s*=\s*[\"']([A-Za-z-]{2,8})[\"']", html_text[:4000])
    return m.group(1) if m else None


def extract_title(html_text: str, fallback: str) -> str:
    if lxml_html is not None:
        try:
            tree = lxml_html.fromstring(html_text)
            for xp in ("//h1[normalize-space()]", "//title[normalize-space()]"):
                found = tree.xpath(xp)
                if found:
                    text = " ".join(found[0].text_content().split())
                    if text:
                        return text[:300]
        except Exception:
            pass
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.DOTALL | re.IGNORECASE)
    if m:
        text = " ".join(re.sub(r"<[^>]+>", " ", m.group(1)).split())
        if text:
            return text[:300]
    return fallback


def extract_main_text(html_text: str, url: str) -> str:
    """Contenu principal en Markdown, sans menus ni pieds de page."""
    if trafilatura is not None:
        try:
            out = trafilatura.extract(
                html_text,
                url=url,
                output_format="markdown",
                include_tables=True,
                include_links=False,
                include_comments=False,
                favor_recall=True,
            )
            if out and out.strip():
                return out.strip()
        except Exception:
            pass
    # Secours : on retire scripts, styles, nav, header, footer, puis on aplatit.
    if lxml_html is not None:
        try:
            tree = lxml_html.fromstring(html_text)
            drop = ("script", "style", "nav", "header", "footer", "noscript", "form")
            for el in list(tree.iter(*drop)):
                parent = el.getparent()
                if parent is not None:
                    parent.remove(el)
            text = tree.text_content()
            return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
        except Exception:
            pass
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html_text)
    text = re.sub(r"<[^>]+>", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_links(html_text: str, base_url: str) -> list[str]:
    """Tous les href absolus de la page."""
    hrefs: list[str] = []
    if lxml_html is not None:
        try:
            tree = lxml_html.fromstring(html_text)
            tree.make_links_absolute(base_url, resolve_base_href=True)
            hrefs = [a.get("href") for a in tree.xpath("//a[@href]")]
        except Exception:
            hrefs = []
    if not hrefs:
        hrefs = [
            urljoin(base_url, m.group(1))
            for m in re.finditer(
                r'<a[^>]+href\s*=\s*["\']([^"\'#]+)["\']', html_text, re.IGNORECASE
            )
        ]
    out, seen = [], set()
    for h in hrefs:
        if not h or h.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
            continue
        h = normalize_url(h)
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def pdf_extract(path: Path, max_pages: int = 80) -> tuple[str, int, str]:
    """Renvoie (texte, nombre de pages, titre). Texte complet, plafonne a max_pages."""
    if PdfReader is None:
        return "", 0, ""
    try:
        reader = PdfReader(str(path))
        n_pages = len(reader.pages)
        parts = [(pg.extract_text() or "") for pg in reader.pages[:max_pages]]
        text = "\n".join(parts).strip()
        title = ""
        try:
            meta = reader.metadata or {}
            title = (meta.get("/Title") or "").strip()
        except Exception:
            title = ""
        if not title:
            # A defaut : la premiere ligne non vide du document.
            for line in text.splitlines():
                line = line.strip()
                if len(line) > 3:
                    title = line[:200]
                    break
        return text, n_pages, title
    except Exception:
        return "", 0, ""


# --------------------------------------------------------------------------
# Sitemap
# --------------------------------------------------------------------------


def _localname(tag: str) -> str:
    """Nom de balise sans son espace de noms."""
    return tag.rsplit("}", 1)[-1].lower()


def parse_sitemap(data: bytes) -> tuple[list[str], list[str]]:
    """Renvoie (urls de pages, urls de sous-sitemaps). Gere le gzip et les namespaces."""
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except Exception:
            return [], []
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError:
        return [], []
    kind = _localname(root.tag)
    locs: list[str] = []
    for child in root:
        for sub in child:
            if _localname(sub.tag) == "loc" and (sub.text or "").strip():
                locs.append(sub.text.strip())
                break
    if kind == "sitemapindex":
        return [], locs
    return locs, []


def _path_parts(url: str) -> list[str]:
    return [p for p in urlparse(url).path.split("/") if p]


def _strip_ext(name: str) -> str:
    for ext in (".html", ".htm", ".php", ".pdf", ".aspx"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def group_by_folder(urls: list[str], segments: int = 2) -> dict[str, int]:
    """Regroupe par dossier. Parlant sur un site a arborescence profonde."""
    groups: dict[str, int] = {}
    for url in urls:
        parts = _path_parts(url)
        dirs = parts[:-1][:segments] if len(parts) > 1 else parts[:segments]
        key = "/" + "/".join(dirs) + "/" if dirs else "/"
        groups[key] = groups.get(key, 0) + 1
    return dict(sorted(groups.items(), key=lambda kv: -kv[1]))


def group_by_slug(urls: list[str]) -> dict[str, int]:
    """Regroupe par premier mot du nom de page : "/fr/formation-*".

    C'est ce regroupement qui parle sur un site TYPO3, ou tout est a plat
    sous /fr/ et ou le theme se lit dans le nom du fichier.
    """
    groups: dict[str, int] = {}
    for url in urls:
        parts = _path_parts(url)
        if not parts:
            groups["/"] = groups.get("/", 0) + 1
            continue
        dirs = parts[:-1][:2]
        last = _strip_ext(parts[-1])
        tokens = last.split("-")
        # On ignore un suffixe purement numerique : formation-1 et formation-2
        # appartiennent au meme ensemble.
        stem = tokens[0] if len(tokens) > 1 else last
        label = f"{stem}-*" if len(tokens) > 1 else stem
        key = "/" + "/".join(dirs + [label]) if dirs else "/" + label
        groups[key] = groups.get(key, 0) + 1
    return dict(sorted(groups.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------
# Ecriture
# --------------------------------------------------------------------------


def yaml_escape(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_markdown(path: Path, rec: DocRecord, body: str) -> None:
    front = "\n".join(
        [
            "---",
            f"doc_id: {yaml_escape(rec.doc_id)}",
            f"title: {yaml_escape(rec.title)}",
            f"url: {yaml_escape(rec.url)}",
            f"lang: {rec.lang}",
            f"fmt: {rec.fmt}",
            f"category: {yaml_escape(rec.category)}",
            f"n_tokens_est: {rec.n_tokens_est}",
            f"scraped_at: {rec.scraped_at}",
            "---",
            "",
        ]
    )
    # trafilatura restitue souvent le H1 de la page : on l'enleve pour ne pas le doubler.
    body = re.sub(r"\A\s*#\s+[^\n]*\n+", "", body)
    path.write_text(f"{front}# {rec.title}\n\n{body.strip()}\n", encoding="utf-8")


CSV_FIELDS = [
    "doc_id",
    "title",
    "url",
    "lang",
    "fmt",
    "category",
    "n_chars",
    "n_tokens_est",
    "n_pages",
    "sha256",
    "scraped_at",
]


def write_documents_csv(path: Path, records: list[DocRecord]) -> None:
    """Fusionne avec le CSV existant : une execution par categorie, un seul index."""
    if not records and not path.exists():
        return
    merged: dict[str, dict] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("doc_id"):
                    merged[row["doc_id"]] = row
    for rec in records:
        merged[rec.doc_id] = {k: str(v) for k, v in asdict(rec).items()}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for doc_id in sorted(merged):
            writer.writerow(merged[doc_id])


# --------------------------------------------------------------------------
# Crawler
# --------------------------------------------------------------------------


class Scraper:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.state_path = self.out_dir / "_scrape_state.json"
        self.csv_path = self.out_dir / "documents.csv"
        self.allow = [re.compile(p, re.IGNORECASE) for p in (args.allow or DEFAULT_ALLOW)]
        self.deny = [re.compile(p, re.IGNORECASE) for p in (args.deny or DEFAULT_DENY)]
        self.pdf_hosts = re.compile(args.pdf_hosts, re.IGNORECASE)
        self.stats = CrawlStats()
        self.records: list[DocRecord] = []
        self.taken_ids: set[str] = set()
        self.visited: set[str] = set()
        self.seen_hashes: dict[str, str] = {}
        self.robots: RobotFileParser | None = None
        verify, self.tls_label = build_verify(args)
        self.client = httpx.Client(
            verify=verify,
            follow_redirects=True,
            timeout=httpx.Timeout(args.timeout),
            headers={
                "User-Agent": args.user_agent,
                "Accept-Language": "fr,en;q=0.8",
            },
        )

    # -- etat -------------------------------------------------------------

    def load_state(self) -> None:
        if self.args.fresh or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.visited = set(data.get("visited", []))
        self.taken_ids = set(data.get("taken_ids", []))
        self.seen_hashes = dict(data.get("seen_hashes", {}))
        print(f"[reprise] {len(self.visited)} URL deja traitees")

    def save_state(self) -> None:
        if self.args.dry_run:
            return
        self.state_path.write_text(
            json.dumps(
                {
                    "visited": sorted(self.visited),
                    "taken_ids": sorted(self.taken_ids),
                    "seen_hashes": self.seen_hashes,
                    "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- robots -----------------------------------------------------------

    def load_robots(self, start_url: str) -> None:
        parsed = urlparse(start_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        try:
            resp = self.client.get(robots_url)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                self.robots = rp
                print(f"[robots] {robots_url} charge")
                delay = rp.crawl_delay(self.args.user_agent) or rp.crawl_delay("*")
                if delay and float(delay) > self.args.delay:
                    self.args.delay = float(delay)
                    print(f"[robots] Crawl-delay impose : {self.args.delay}s")
            else:
                print(f"[robots] {robots_url} -> HTTP {resp.status_code}, aucune regle")
        except Exception as exc:
            msg = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" in msg or "SSL" in msg.upper():
                print()
                print("[tls] Le certificat du site n'a pas pu etre valide depuis cette machine.")
                print(f"      {msg[:160]}")
                print()
                print("      Ce n'est pas un refus du site : c'est la chaine de certification")
                print(f"      vue par Python. Magasin utilise ici : {self.tls_label}.")
                print()
                if "truststore" in self.tls_label:
                    # Le magasin de Windows a deja ete consulte et il a refuse.
                    print("      Le magasin de Windows a deja ete consulte, sans succes.")
                    print("      Exporte la chaine depuis ton navigateur (cadenas > certificat >")
                    print("      details > exporter en PEM, chaine complete), puis :")
                    print("          ... --ca-bundle C:\\chemin\\chaine.pem")
                elif self.args.ca_bundle:
                    print("      Le paquet fourni ne valide pas ce certificat. Verifie qu'il")
                    print("      contient bien toute la chaine, racine comprise.")
                else:
                    print(
                        "      Remede le plus frequent sous Windows (antivirus ou reseau filtrant) :"
                    )
                    print("          uv add truststore")
                    print("          puis relance exactement la meme commande.")
                print()
                print("      Pour identifier la cause exacte :")
                print("          uv run python scripts/diag_tls.py")
                sys.exit(4)
            print(f"[robots] illisible ({exc}) : par prudence, on s'arrete.")
            print("         Verifie le fichier a la main avant de relancer.")
            sys.exit(2)

    def allowed_by_robots(self, url: str) -> bool:
        if self.robots is None:
            return True
        # robotparser retombe seul sur l'entree User-agent: * si l'agent n'est pas liste.
        return self.robots.can_fetch(self.args.user_agent, url)

    # -- reseau -----------------------------------------------------------

    def fetch(self, url: str):
        """GET avec reprise exponentielle sur 429 et 5xx."""
        wait = self.args.delay
        for attempt in range(1, self.args.retries + 1):
            try:
                resp = self.client.get(url)
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("retry-after")
                    pause = float(retry_after) if (retry_after or "").isdigit() else wait
                    print(
                        f"  HTTP {resp.status_code}, attente {pause:.0f}s "
                        f"(essai {attempt}/{self.args.retries})"
                    )
                    time.sleep(min(pause, 120))
                    wait = min(wait * 2, 120)
                    continue
                return resp
            except httpx.HTTPError as exc:
                print(f"  erreur reseau : {exc} (essai {attempt}/{self.args.retries})")
                time.sleep(min(wait, 60))
                wait = min(wait * 2, 120)
        return None

    # -- traitement d'une page -------------------------------------------

    def handle_page(self, url: str, depth: int, frontier: list[tuple[str, int]]) -> None:
        resp = self.fetch(url)
        if resp is None or resp.status_code >= 400:
            code = "?" if resp is None else resp.status_code
            self.stats.pages_failed += 1
            self.stats.errors.append(f"page {url} -> HTTP {code}")
            print(f"  echec HTTP {code}")
            return

        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype == "application/pdf":
            self.handle_pdf(url, prefetched=resp.content)
            return
        if ctype and not any(ctype.startswith(t) for t in TEXT_HTML):
            print(f"  ignore (content-type {ctype})")
            return

        html_text = resp.text
        body = extract_main_text(html_text, url)
        title = extract_title(html_text, fallback=url)
        html_lang = extract_html_lang(html_text)

        # Nouveaux liens, meme si la page est trop maigre pour etre gardee.
        if depth < self.args.max_depth:
            for link in extract_links(html_text, url):
                if is_pdf_url(link):
                    if self.args.with_pdf and self.pdf_hosts.search(urlparse(link).netloc):
                        frontier.append((link, depth + 1))
                elif self.in_page_scope(link):
                    frontier.append((link, depth + 1))

        if len(body) < self.args.min_chars:
            self.stats.pages_skipped_thin += 1
            print(f"  ignore ({len(body)} caracteres < {self.args.min_chars})")
            return

        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if digest in self.seen_hashes:
            self.stats.already_present += 1
            print(f"  doublon de contenu (identique a {self.seen_hashes[digest]})")
            return

        doc_id = url_to_doc_id(url, self.taken_ids)
        lang = detect_lang(body, url, html_lang, self.args.lang_hint)
        rec = DocRecord(
            doc_id=doc_id,
            title=title,
            url=url,
            lang=lang,
            fmt="md",
            category=self.args.category,
            n_chars=len(body),
            n_tokens_est=count_tokens(body),
            n_pages=0,
            sha256=digest,
            scraped_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        if not self.args.dry_run:
            write_markdown(self.out_dir / f"{doc_id}.md", rec, body)
        self.seen_hashes[digest] = doc_id
        self.records.append(rec)
        self.stats.pages_ok += 1
        print(f"  OK  {doc_id}.md  [{lang}]  {rec.n_tokens_est} tokens  {title[:60]}")

    # -- traitement d'un PDF ---------------------------------------------

    def handle_pdf(self, url: str, prefetched: bytes | None = None) -> None:
        max_bytes = int(self.args.max_pdf_mb * 1024 * 1024)
        doc_id = url_to_doc_id(url, self.taken_ids)
        target = self.out_dir / f"{doc_id}.pdf"

        if self.args.dry_run:
            print(f"  PDF (simulation) {doc_id}.pdf  <- {url}")
            self.stats.pdfs_ok += 1
            return

        if prefetched is not None:
            if len(prefetched) > max_bytes:
                self.stats.pdfs_too_large += 1
                print(f"  PDF ignore ({len(prefetched) / 1e6:.1f} Mo > {self.args.max_pdf_mb} Mo)")
                return
            target.write_bytes(prefetched)
        else:
            total = 0
            try:
                with self.client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        self.stats.pdfs_failed += 1
                        self.stats.errors.append(f"pdf {url} -> HTTP {resp.status_code}")
                        print(f"  PDF echec HTTP {resp.status_code}")
                        return
                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > max_bytes:
                        self.stats.pdfs_too_large += 1
                        print(f"  PDF ignore ({int(declared) / 1e6:.1f} Mo annonces)")
                        return
                    with target.open("wb") as fh:
                        for chunk in resp.iter_bytes(65536):
                            total += len(chunk)
                            if total > max_bytes:
                                fh.close()
                                target.unlink(missing_ok=True)
                                self.stats.pdfs_too_large += 1
                                print(f"  PDF ignore (> {self.args.max_pdf_mb} Mo)")
                                return
                            fh.write(chunk)
            except httpx.HTTPError as exc:
                self.stats.pdfs_failed += 1
                self.stats.errors.append(f"pdf {url} -> {exc}")
                target.unlink(missing_ok=True)
                print(f"  PDF echec : {exc}")
                return

        data = target.read_bytes()
        if not data.startswith(b"%PDF"):
            target.unlink(missing_ok=True)
            self.stats.pdfs_failed += 1
            self.stats.errors.append(f"pdf {url} -> ce n'est pas un PDF")
            print("  PDF echec : signature %PDF absente")
            return

        digest = hashlib.sha256(data).hexdigest()
        if digest in self.seen_hashes:
            target.unlink(missing_ok=True)
            self.stats.already_present += 1
            print(f"  PDF doublon (identique a {self.seen_hashes[digest]})")
            return

        text, n_pages, pdf_title = pdf_extract(target)
        if len(text) < 100 and n_pages:
            print("  ATTENTION : peu ou pas de texte extrait -> PDF probablement scanne")
            self.stats.errors.append(f"pdf {url} -> sans couche texte (scan ?)")

        rec = DocRecord(
            doc_id=doc_id,
            title=pdf_title or doc_id.replace("-", " ").title(),
            url=url,
            lang=detect_lang(text, url, None, self.args.lang_hint),
            fmt="pdf",
            category=self.args.category,
            n_chars=len(text),
            n_tokens_est=count_tokens(text),
            n_pages=n_pages,
            sha256=digest,
            scraped_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.seen_hashes[digest] = doc_id
        self.records.append(rec)
        self.stats.pdfs_ok += 1
        print(f"  OK  {doc_id}.pdf  [{rec.lang}]  {n_pages} pages  {len(data) / 1e6:.2f} Mo")

    # -- sitemap ----------------------------------------------------------

    def sitemap_roots(self, start_url: str) -> list[str]:
        """Ou chercher le sitemap : option explicite, puis robots.txt, puis defaut."""
        if self.args.sitemap:
            return [self.args.sitemap]
        if self.robots is not None:
            declared = self.robots.site_maps()
            if declared:
                return list(declared)
        parsed = urlparse(start_url)
        return [f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"]

    def collect_sitemap_urls(self, start_url: str) -> list[str]:
        """Parcourt le sitemap et ses sous-sitemaps, renvoie toutes les URLs listees."""
        queue = [(u, 0) for u in self.sitemap_roots(start_url)]
        seen_maps: set[str] = set()
        urls: list[str] = []
        while queue and len(seen_maps) < self.args.max_sitemaps:
            sm_url, depth = queue.pop(0)
            sm_url = normalize_url(sm_url)
            if sm_url in seen_maps or depth > 3:
                continue
            seen_maps.add(sm_url)
            resp = self.fetch(sm_url)
            if resp is None or resp.status_code >= 400:
                code = "?" if resp is None else resp.status_code
                print(f"[sitemap] {sm_url} -> HTTP {code}")
                continue
            pages, children = parse_sitemap(resp.content)
            if pages:
                print(f"[sitemap] {sm_url} : {len(pages)} URL")
                urls.extend(pages)
            if children:
                print(f"[sitemap] {sm_url} : index de {len(children)} sous-sitemaps")
                queue.extend((c, depth + 1) for c in children)
            time.sleep(self.args.delay)
        # Dedoublonnage en conservant l'ordre
        out, seen = [], set()
        for u in urls:
            u = normalize_url(u)
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def report_sitemap(self, urls: list[str]) -> None:
        """Inventaire lisible : c'est ce qui remplace les motifs devines."""
        print()
        print("=" * 68)
        print(f"INVENTAIRE DU SITEMAP : {len(urls)} URL au total")
        print("=" * 68)
        print()
        by_slug = group_by_slug(urls)
        by_folder = group_by_folder(urls)

        print("Par prefixe de nom de page (lecture utile sur un site TYPO3)")
        for prefix, count in list(by_slug.items())[:30]:
            print(f"  {count:>5}  {prefix}")
        if len(by_slug) > 30:
            print(f"        ... et {len(by_slug) - 30} autres prefixes")
        print()

        if len(by_folder) > 1:
            print("Par dossier")
            for prefix, count in list(by_folder.items())[:15]:
                print(f"  {count:>5}  {prefix}")
            print()

        kept = [u for u in urls if self.in_page_scope(u)]
        pdfs = [u for u in urls if is_pdf_url(u)]
        print(f"Retenues par les motifs --allow/--deny actuels : {len(kept)}")
        print(f"PDF listes directement dans le sitemap : {len(pdfs)}")
        print()
        if kept:
            print("Echantillon des URL retenues :")
            step = max(1, len(kept) // 20)
            for u in kept[::step][:20]:
                print(f"  {u}")
        else:
            print("Aucune URL retenue. Elargis avec --allow, par exemple :")
            for prefix in list(by_slug)[:8]:
                print(f'  --allow "{prefix.rstrip("*").rstrip("-")}"')
        print()
        print("Pour cibler un theme, reprends un prefixe ci-dessus, sans l'etoile :")
        print('  --allow "/fr/formation" --allow "/en/stud"')
        print()
        print("Pour lancer la collecte sur ces URL :")
        print("  ... --from-sitemap --category formations --max-pages 40")
        print()
        print("=" * 68)

    # -- portee -----------------------------------------------------------

    def in_page_scope(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if self.args.same_host and parsed.netloc != self.start_host:
            return False
        if matches_any(url, self.deny):
            return False
        return matches_any(url, self.allow)

    # -- boucle principale ------------------------------------------------

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[tls] {self.tls_label}")
        start = normalize_url(self.args.start_url)
        self.start_host = urlparse(start).netloc
        self.load_state()

        if not self.args.no_robots_check:
            self.load_robots(start)
            if not self.allowed_by_robots(start):
                print(f"\n[robots] {start} est interdit au crawl par robots.txt.")
                print("Le script s'arrete : c'est une regle du site, pas une option du script.")
                print("Demande l'autorisation a la DSI de l'UHA, ou utilise une autre source.")
                sys.exit(3)

        frontier: list[tuple[str, int]] = []

        if self.args.list_sitemap or self.args.from_sitemap:
            sitemap_urls = self.collect_sitemap_urls(start)
            if not sitemap_urls:
                print("[sitemap] aucune URL trouvee. On repart du crawl classique.")
            if self.args.list_sitemap:
                self.report_sitemap(sitemap_urls)
                self.client.close()
                return
            kept = [
                u
                for u in sitemap_urls
                if self.in_page_scope(u)
                or (
                    self.args.with_pdf
                    and is_pdf_url(u)
                    and self.pdf_hosts.search(urlparse(u).netloc)
                )
            ]
            print(f"[sitemap] {len(kept)} URL retenues sur {len(sitemap_urls)}")
            frontier.extend((u, 0) for u in kept)

        if not self.args.from_sitemap or not frontier:
            frontier.insert(0, (start, 0))
        for extra in self.args.also_url or []:
            frontier.append((normalize_url(extra), 0))

        processed = 0
        while frontier and processed < self.args.max_pages:
            url, depth = frontier.pop(0)
            url = normalize_url(url)
            if url in self.visited:
                continue
            self.visited.add(url)

            if not self.allowed_by_robots(url):
                self.stats.blocked_by_robots += 1
                print(f"[{processed + 1}] {url}\n  interdit par robots.txt")
                continue

            processed += 1
            print(f"[{processed}/{self.args.max_pages}] {url}")
            if is_pdf_url(url):
                if self.args.with_pdf:
                    self.handle_pdf(url)
            else:
                self.handle_page(url, depth, frontier)

            if processed % 10 == 0:
                self.save_state()
                if self.records and not self.args.dry_run:
                    write_documents_csv(self.csv_path, self.records)
            time.sleep(self.args.delay)

        self.save_state()
        if not self.args.dry_run:
            write_documents_csv(self.csv_path, self.records)
            self.write_report()
        self.client.close()
        self.summary(frontier)

    # -- sorties finales --------------------------------------------------

    def read_corpus(self) -> list[dict]:
        """Corpus complet tel qu'il est sur le disque, toutes executions confondues."""
        if not self.csv_path.exists():
            return [{k: str(v) for k, v in asdict(r).items()} for r in self.records]
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            return [row for row in csv.DictReader(f) if row.get("doc_id")]

    def write_report(self) -> None:
        s = self.stats
        corpus = self.read_corpus()

        def as_int(row: dict, key: str) -> int:
            try:
                return int(row.get(key) or 0)
            except ValueError:
                return 0

        by_lang: dict[str, int] = {}
        by_fmt: dict[str, int] = {}
        by_cat: dict[str, int] = {}
        total_tokens = 0
        for row in corpus:
            by_lang[row.get("lang", "?")] = by_lang.get(row.get("lang", "?"), 0) + 1
            by_fmt[row.get("fmt", "?")] = by_fmt.get(row.get("fmt", "?"), 0) + 1
            by_cat[row.get("category", "?")] = by_cat.get(row.get("category", "?"), 0) + 1
            total_tokens += as_int(row, "n_tokens_est")
        run_tokens = sum(r.n_tokens_est for r in self.records)

        lines = [
            "# Rapport de collecte",
            "",
            f"- Date : {datetime.now(UTC).isoformat(timespec='seconds')}",
            f"- URL de depart : {self.args.start_url}",
            f"- Categorie de cette execution : {self.args.category}",
            "",
            "## Corpus total (toutes executions)",
            "",
            f"- Documents : {len(corpus)}",
            f"- Tokens estimes (cl100k_base) : {total_tokens}",
            (
                f"- Chunks attendus a {self.args.chunk_tokens}/"
                f"{self.args.chunk_overlap} tokens : ~"
                f"{max(0, total_tokens // max(1, self.args.chunk_tokens - self.args.chunk_overlap))}"
            ),
            f"- Par langue : {by_lang or '-'}",
            f"- Par format : {by_fmt or '-'}",
            f"- Par categorie : {by_cat or '-'}",
            "",
            "Rappel de la cible du plan : 30 a 40 documents, 80 000 a 100 000 tokens,",
            "au moins un tiers en anglais, sur 3 a 4 themes qui se recoupent.",
            "",
            "## Cette execution",
            "",
            f"- Pages web gardees : {s.pages_ok}",
            f"- PDF telecharges : {s.pdfs_ok}",
            f"- Tokens ajoutes : {run_tokens}",
            f"- Pages trop maigres ignorees : {s.pages_skipped_thin}",
            f"- Doublons de contenu : {s.already_present}",
            f"- Pages en echec : {s.pages_failed}",
            f"- PDF en echec : {s.pdfs_failed}",
            f"- PDF trop volumineux : {s.pdfs_too_large}",
            f"- Bloques par robots.txt : {s.blocked_by_robots}",
            "",
        ]
        if s.errors:
            lines += ["## Details des erreurs de cette execution", ""]
            lines += [f"- {e}" for e in s.errors[:50]]
            lines.append("")
        lines += [
            "## Documents du corpus",
            "",
            "| doc_id | lang | fmt | categorie | tokens | titre |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in sorted(corpus, key=lambda r: r.get("doc_id", "")):
            title = (row.get("title") or "").replace("|", "/")[:70]
            lines.append(
                f"| {row.get('doc_id', '')} | {row.get('lang', '')} | {row.get('fmt', '')} "
                f"| {row.get('category', '')} | {as_int(row, 'n_tokens_est')} | {title} |"
            )
        (self.out_dir / "_scrape_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def summary(self, frontier: list) -> None:
        s = self.stats
        tokens = sum(r.n_tokens_est for r in self.records)
        print("\n" + "=" * 62)
        print(
            f"Pages gardees : {s.pages_ok}   PDF : {s.pdfs_ok}   "
            f"Tokens estimes : {tokens}  (~{max(0, tokens // 1100)} chunks)"
        )
        print(
            f"Ignorees (maigres) : {s.pages_skipped_thin}   Doublons : {s.already_present}   "
            f"Echecs : {s.pages_failed + s.pdfs_failed}   Robots : {s.blocked_by_robots}"
        )
        if frontier:
            print(
                f"File d'attente non vide : {len(frontier)} URL. "
                f"Relance avec --max-pages plus grand pour continuer."
            )
        if not self.args.dry_run:
            print(f"Sortie : {self.out_dir.resolve()}")
            print(f"Index  : {self.csv_path.resolve()}")
            print(f"Rapport: {(self.out_dir / '_scrape_report.md').resolve()}")
        print("=" * 62)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collecte du corpus UHA (pages web -> Markdown, PDF -> fichiers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start-url", default=DEFAULT_START, help="URL de depart")
    p.add_argument("--also-url", action="append", help="URL supplementaire a visiter (repetable)")
    p.add_argument("--out-dir", default="data/raw", help="dossier de sortie")
    p.add_argument("--category", default="formations", help="theme, ecrit dans documents.csv")
    p.add_argument(
        "--lang-hint",
        default="fr",
        choices=["fr", "en"],
        help="langue par defaut si la detection echoue",
    )
    p.add_argument("--max-pages", type=int, default=40, help="nombre d'URL traitees au maximum")
    p.add_argument("--max-depth", type=int, default=2, help="profondeur de suivi des liens")
    p.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="pause entre deux requetes, en secondes",
    )
    p.add_argument("--timeout", type=float, default=30.0, help="delai d'attente HTTP")
    p.add_argument("--retries", type=int, default=4, help="tentatives en cas de 429 ou 5xx")
    p.add_argument(
        "--min-chars",
        type=int,
        default=400,
        help="en dessous, la page est jugee trop maigre",
    )
    p.add_argument("--max-pdf-mb", type=float, default=25.0, help="taille maximale d'un PDF")
    p.add_argument(
        "--chunk-tokens",
        type=int,
        default=1200,
        help="taille de chunk, pour l'estimation du rapport (config.yaml)",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="recouvrement, pour l'estimation du rapport (config.yaml)",
    )
    p.add_argument(
        "--no-pdf",
        dest="with_pdf",
        action="store_false",
        help="ne pas telecharger les PDF",
    )
    p.add_argument(
        "--pdf-hosts",
        default=DEFAULT_PDF_HOSTS,
        help="expression reguliere des hotes autorises pour les PDF",
    )
    p.add_argument("--allow", action="append", help="motif d'URL a suivre (remplace les defauts)")
    p.add_argument("--deny", action="append", help="motif d'URL a exclure (remplace les defauts)")
    p.add_argument(
        "--any-host",
        dest="same_host",
        action="store_false",
        help="autoriser les pages hors du domaine de depart",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignorer l'etat precedent et tout recollecter",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="n'ecrit rien, affiche ce qui serait collecte",
    )
    p.add_argument(
        "--no-robots-check", action="store_true", help=argparse.SUPPRESS
    )  # usage interne : tests sur serveur local
    p.add_argument(
        "--list-sitemap",
        action="store_true",
        help="afficher l'inventaire des URL du sitemap, puis s'arreter",
    )
    p.add_argument(
        "--from-sitemap",
        action="store_true",
        help="partir des URL du sitemap au lieu de suivre les liens",
    )
    p.add_argument(
        "--sitemap",
        default=None,
        help="URL du sitemap (par defaut : celle de robots.txt, sinon /sitemap.xml)",
    )
    p.add_argument(
        "--max-sitemaps",
        type=int,
        default=50,
        help="nombre maximal de sous-sitemaps a lire",
    )
    p.add_argument(
        "--ssl",
        choices=["auto", "certifi", "system"],
        default="auto",
        help="magasin de certificats : auto (systeme si truststore est la, "
        "sinon certifi), certifi, ou system",
    )
    p.add_argument(
        "--ca-bundle",
        default=None,
        help="chemin d'un fichier PEM de certificats a utiliser pour la validation",
    )
    p.add_argument("--user-agent", default=USER_AGENT, help="en-tete User-Agent")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if trafilatura is None:
        print("[info] trafilatura absent : extraction de secours, qualite moindre.")
    if detect is None:
        print("[info] langdetect absent : detection de langue par URL seulement.")
    Scraper(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
