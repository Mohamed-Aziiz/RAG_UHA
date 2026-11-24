"""Interface de démo : une question, deux méthodes de recherche, côte à côte.

    uv run streamlit run app/streamlit_app.py

Identité : le logo de l'UHA est formé de deux arcs, bleu (#004995) et cyan (#009ee2). Chaque
méthode prend l'un des deux bleus : bleu pour le RAG classique, cyan pour le GraphRAG. Thème
sombre, sur le bleu nuit de l'UHA ; le logo est affiché dans sa version claire (logo_uha_sombre.svg).
Les réponses sont composées en Literata (police de lecture), l'interface en Archivo.

Mode hors ligne : si l'API est indisponible, les réponses enregistrées dans
`data/demo_cache.json` (créé par scripts/record_demo.py) sont affichées à la place.
"""

from __future__ import annotations

import base64
import html
import json
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_uha.config import load_settings
from rag_uha.loaders import read_documents_csv
from rag_uha.prompts import is_refusal
from rag_uha.types import Answer

settings = load_settings()
st.set_page_config(
    page_title=settings.app.name,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PIPELINES = {
    "rag": {
        "titre": "RAG classique",
        "methode": "Retrouve les passages les plus proches de la question.",
    },
    "graphrag": {
        "titre": "GraphRAG",
        "methode": "Suit les liens entre les notions extraites des documents.",
    },
}
CHOIX = {"Les deux": ["rag", "graphrag"], "RAG classique": ["rag"], "GraphRAG": ["graphrag"]}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600&display=swap');

    :root {
      --fond: #0b1a30;            /* bleu nuit, le bleu UHA #004995 assombri */
      --uha-bleu: #4d94e8;        /* arc bleu du logo, eclairci pour le fond sombre : RAG classique */
      --uha-cyan: #19b8f0;        /* arc cyan du logo : GraphRAG */
      --uha-cyan-texte: #3fc6f5;
      --encre: #e8eef7;
      --gris: #9aabc4;
      --filet: #22395e;
      --lecture: 'Literata', Georgia, 'Times New Roman', serif;
    }
    .block-container { max-width: 1180px; padding-top: 4.2rem; padding-bottom: 4rem; }
    [data-testid="stHeader"] { background: transparent; }  /* la barre du haut masquait le logo */

    /* En-tete */
    .hero .logo { display: block; height: 54px; max-width: 100%; margin-bottom: 2.2rem; }
    .hero h1 { font-size: clamp(2.1rem, 4.4vw, 3.3rem); line-height: 1.06; font-weight: 800;
               letter-spacing: -0.025em; color: var(--encre);
               margin: 0 0 1rem; padding: 0; }
    .hero p { font-size: 1.1rem; line-height: 1.55; color: var(--gris); max-width: 60ch; margin: 0 0 2rem; }

    /* La question posee devient le titre des resultats */
    .question-posee { font-family: var(--lecture); font-size: clamp(1.45rem, 2.7vw, 2.05rem);
                      line-height: 1.28; color: var(--encre); max-width: 42ch;
                      margin: 2.6rem 0 1.8rem; }

    /* Deux colonnes, deux arcs : un filet epais a la couleur de chaque methode */
    .st-key-col_rag, .st-key-col_graphrag { position: relative; padding-left: 1.35rem; }
    .st-key-col_rag::before, .st-key-col_graphrag::before {
      content: ""; position: absolute; left: 0; top: .2rem; bottom: .2rem; width: 6px;
      border-radius: 3px; transform-origin: top;
      animation: trace .75s cubic-bezier(.2, .7, .2, 1) both;
    }
    .st-key-col_rag::before { background: var(--uha-bleu); }
    .st-key-col_graphrag::before { background: var(--uha-cyan); animation-delay: .14s; }
    @keyframes trace { from { transform: scaleY(0); } to { transform: scaleY(1); } }
    @media (prefers-reduced-motion: reduce) {
      .st-key-col_rag::before, .st-key-col_graphrag::before { animation: none; }
    }

    .entete h2 { font-size: 1.4rem; font-weight: 700; margin: 0; padding: 0; letter-spacing: -0.01em; }
    .st-key-col_rag .entete h2 { color: var(--uha-bleu); }
    .st-key-col_graphrag .entete h2 { color: var(--uha-cyan-texte); }
    .entete p { color: var(--gris); font-size: .95rem; margin: .25rem 0 1.1rem; }

    /* Le texte des reponses se lit : Literata */
    .st-key-rep_rag p, .st-key-rep_rag li, .st-key-rep_graphrag p, .st-key-rep_graphrag li {
      font-family: var(--lecture); font-size: 1.05rem; line-height: 1.65; color: var(--encre);
    }

    /* Appels de note : un numero dans la couleur de la methode, la page source au survol */
    .note { display: inline-block; min-width: 1.25em; margin-left: .15em; padding: 0 .3em;
            font-family: 'Archivo', sans-serif; font-size: .68em; font-weight: 700; line-height: 1.5;
            text-align: center; vertical-align: .4em; border-radius: 3px; text-decoration: none !important;
            color: var(--fond) !important; background: var(--uha-bleu); }
    .st-key-col_graphrag .note { background: var(--uha-cyan); }
    .note:hover, .note:focus-visible { outline: 2px solid var(--encre); outline-offset: 1px; }
    ol.cites { list-style: none; margin: .4rem 0 0; padding: 0; font-size: .9rem; }
    ol.cites li { display: flex; gap: .6rem; margin: 0 0 .3rem; padding: 0; color: var(--gris); }
    ol.cites li span { flex: none; min-width: 1.4em; font-weight: 700; color: var(--uha-bleu);
                       font-variant-numeric: tabular-nums; }
    .st-key-col_graphrag ol.cites li span { color: var(--uha-cyan-texte); }

    /* Trois faits par reponse */
    .faits { display: flex; flex-wrap: wrap; gap: .5rem 2rem; margin: 1.2rem 0 .7rem;
             padding-top: .85rem; border-top: 1px solid var(--filet); }
    .faits div { display: flex; flex-direction: column; }
    .faits dt { font-size: .8rem; color: var(--gris); }
    .faits dd { margin: 0; font-size: 1.1rem; font-weight: 700; color: var(--encre);
                font-variant-numeric: tabular-nums; }

    /* Le verdict : une phrase, marquee des deux couleurs du logo */
    .verdict { font-size: 1.15rem; line-height: 1.55; font-weight: 500; color: var(--encre);
               max-width: 70ch; margin: 2.6rem 0 0; }
    .verdict::before { content: ""; display: inline-block; width: 30px; height: 6px;
                       margin-right: .75rem; vertical-align: .18em; border-radius: 3px;
                       background: linear-gradient(90deg, var(--uha-bleu) 50%, var(--uha-cyan) 50%); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Chargement des index…")
def get_pipeline(name: str):
    from rag_uha.pipelines.base import build_pipeline

    return build_pipeline(name, settings)


@st.cache_data
def demo_cache() -> dict:
    p = settings.paths.demo_cache_file
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_resource
def graph():
    from rag_uha.indexing.graph_index import load_graph

    return load_graph(settings)


@st.cache_data
def documents() -> dict[str, dict[str, str]]:
    """Titre et adresse de chaque document, d'apres l'index du scraping (data/raw/documents.csv)."""
    return read_documents_csv(settings.paths.documents_csv)


def ask(name: str, question: str, offline: bool) -> Answer:
    if offline:
        cached = demo_cache().get(name, {}).get(question.strip())
        if cached:
            return Answer.from_dict(cached)
        raise RuntimeError("cette question n'a pas de réponse enregistrée pour le mode hors ligne")
    return get_pipeline(name).answer(question)


# ------------------------------------------------------------------ mise en forme
def logo_html() -> str:
    p = settings.app.logo_path
    sombre = p.with_name(f"{p.stem}_sombre{p.suffix}")  # version claire du logo, pour le fond sombre
    p = sombre if sombre.exists() else p
    if not p.exists():
        return ""
    mime = "image/svg+xml" if p.suffix.lower() == ".svg" else "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img class="logo" src="data:{mime};base64,{data}" alt="Université de Haute-Alsace">'


def nombre(n: float, decimales: int = 0) -> str:
    txt = f"{n:,.{decimales}f}".replace(",", " ").replace(".", ",")
    return txt


def fois(r: float) -> str:
    return nombre(r) if r >= 10 else nombre(r, 1)


_CITATION = re.compile(r"\[([^\[\]\n]{3,400})\]")
_DOC_ID = re.compile(r"^[a-z0-9][a-z0-9._-]+$")


def titre_doc(doc_id: str) -> str:
    t = documents().get(doc_id, {}).get("title", "")
    t = re.sub(r"\s*[-|]\s*(UHA|CMS Java Open Source)\s*$", "", t).strip()
    return t if t and len(t) > 6 else doc_id


def citer(texte: str) -> tuple[str, list[str]]:
    """Remplace « [doc-id] » par un appel de note numerote ; renvoie le texte et les documents cites."""
    cites: list[str] = []

    def numero(doc_id: str) -> str:
        if doc_id not in cites:
            cites.append(doc_id)
        n = cites.index(doc_id) + 1
        url = documents().get(doc_id, {}).get("url", "")
        titre = html.escape(titre_doc(doc_id), quote=True)
        if url:
            return f'<a class="note" href="{html.escape(url, quote=True)}" title="{titre}" target="_blank">{n}</a>'
        return f'<span class="note" title="{titre}">{n}</span>'

    def remplacer(m: re.Match) -> str:
        ids = [x.strip() for x in re.split(r"[,;]", m.group(1).replace("source:", ""))]
        if not ids or not all(_DOC_ID.match(i) for i in ids):
            return m.group(0)
        return "".join(numero(i) for i in ids)

    texte = texte.replace("<", "&lt;").replace(">", "&gt;")
    return _CITATION.sub(remplacer, texte), cites


def liste_sources(cites: list[str]) -> str:
    lignes = []
    for n, doc_id in enumerate(cites, 1):
        url = documents().get(doc_id, {}).get("url", "")
        titre = html.escape(titre_doc(doc_id))
        lien = f'<a href="{html.escape(url, quote=True)}" target="_blank">{titre}</a>' if url else titre
        lignes.append(f"<li><span>{n}</span>{lien}</li>")
    return f'<ol class="cites">{"".join(lignes)}</ol>' if lignes else ""


def faits(ans: Answer) -> str:
    items = [
        ("Temps de réponse", f"{nombre(ans.latency_s, 1)} s"),
        ("Tokens consommés", nombre(ans.usage.total_tokens)),
        ("Passages lus", nombre(len(ans.contexts))),
    ]
    corps = "".join(f"<div><dt>{a}</dt><dd>{b}</dd></div>" for a, b in items)
    return f'<dl class="faits">{corps}</dl>'


def verdict(rag: Answer, gr: Answer) -> str:
    """Une phrase qui dit ce que la comparaison montre sur cette question."""
    phrases = []
    r_ref, g_ref = is_refusal(rag.text), is_refusal(gr.text)
    if r_ref != g_ref:
        seul = "le GraphRAG" if r_ref else "le RAG classique"
        phrases.append(f"Sur cette question, seul {seul} a trouvé une réponse dans les documents.")
    elif r_ref and g_ref:
        phrases.append("Aucune des deux méthodes n'a trouvé la réponse dans les documents.")

    t = gr.latency_s / rag.latency_s if rag.latency_s else 1
    k = gr.usage.total_tokens / rag.usage.total_tokens if rag.usage.total_tokens else 1
    if t >= 1.5:
        temps = f"a mis {fois(t)} fois plus de temps"
    elif t <= 1 / 1.5:
        temps = f"a été {fois(1 / t)} fois plus rapide"
    else:
        temps = "a répondu aussi vite"
    if k >= 1.5:
        cout = f"consommé {fois(k)} fois plus de tokens"
    elif k <= 1 / 1.5:
        cout = f"consommé {fois(1 / k)} fois moins de tokens"
    else:
        cout = "consommé autant de tokens"
    debut = "Le GraphRAG" if phrases else "Sur cette question, le GraphRAG"
    phrases.append(f"{debut} {temps} que le RAG classique et a {cout}.")

    communs = set(rag.sources) & set(gr.sources)
    if communs:
        n = len(communs)
        phrases.append(
            f"Les deux ont puisé dans {n} document{'s' if n > 1 else ''} commun{'s' if n > 1 else ''}."
        )
    else:
        phrases.append("Ils n'ont puisé dans aucun document commun.")
    return f'<div class="verdict">{html.escape(" ".join(phrases))}</div>'


def colonne(name: str, res: Answer | str) -> None:
    meta = PIPELINES[name]
    with st.container(key=f"col_{name}"):
        st.markdown(
            f'<div class="entete"><h2>{meta["titre"]}</h2><p>{meta["methode"]}</p></div>',
            unsafe_allow_html=True,
        )
        if isinstance(res, str):
            st.error(
                f"Pas de réponse : {res}. Pendant la démo, activez le mode hors ligne "
                "dans le panneau latéral pour rejouer les réponses enregistrées."
            )
            return
        texte, cites = citer(res.text)
        with st.container(key=f"rep_{name}"):
            st.markdown(texte, unsafe_allow_html=True)
        if cites:
            st.markdown(liste_sources(cites), unsafe_allow_html=True)
        st.markdown(faits(res), unsafe_allow_html=True)

        with st.expander(f"Passages utilisés ({len(res.contexts)})"):
            for rc in res.contexts:
                st.markdown(f"**{rc.rank}. {titre_doc(rc.chunk.doc_id)}**")
                st.caption(rc.chunk.text[:500] + ("…" if len(rc.chunk.text) > 500 else ""))

        if name == "graphrag":
            ents = res.extra.get("entities", [])
            rels = res.extra.get("relations", [])
            with st.expander(f"Notions du graphe ({len(ents)} entités, {len(rels)} relations)"):
                kw = res.extra.get("keywords", {})
                mots = kw.get("high_level", []) + kw.get("low_level", []) if kw else []
                if mots:
                    st.caption(f"Mots-clés tirés de la question : {', '.join(mots)}")
                st.write(", ".join(f"{e['name']} *({e['type']})*" for e in ents[:25]))
                if ents and st.button("Voir le sous-graphe", key="sous_graphe"):
                    from rag_uha.viz import subgraph_around, to_html

                    sg = subgraph_around(graph(), [e["name"] for e in ents[:10]], hops=1)
                    page = to_html(sg, ROOT / "data" / "subgraph.html")
                    st.components.v1.html(page.read_text(encoding="utf-8"), height=520, scrolling=True)


# ------------------------------------------------------------------ panneau lateral
with st.sidebar:
    st.subheader("Réglages de la démo")
    offline = st.toggle("Rejouer les réponses enregistrées (hors ligne)", value=False)
    if offline and demo_cache():
        st.caption("Questions enregistrées :")
        for q in list(next(iter(demo_cache().values()), {}).keys())[:8]:
            st.caption(q)
    st.divider()
    st.caption(f"Génération : {settings.llm_query.model}")
    st.caption(f"Embeddings : {settings.embeddings.model}")
    st.caption(f"GraphRAG : mode {settings.graphrag.mode}")

# ------------------------------------------------------------------ page
st.markdown(
    f"""
    <div class="hero">
      {logo_html()}
      <h1>Deux façons de lire<br>les documents de l'UHA</h1>
      <p>Le RAG classique retrouve les passages les plus proches de votre question. Le GraphRAG
      suit les liens entre les notions qu'il a extraites des documents. Posez une question,
      en français ou en anglais, et comparez leurs réponses.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

defaut = {"rag": "RAG classique", "graphrag": "GraphRAG"}.get(settings.app.default_mode, "Les deux")
with st.form("question_form", border=False):
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    question = c1.text_input(
        "Votre question",
        placeholder="Quel âge faut-il avoir pour signer un contrat d'apprentissage ?",
    )
    envoyer = c2.form_submit_button("Demander", type="primary")
    methodes = st.segmented_control(
        "Méthodes à comparer", list(CHOIX), default=defaut, required=True
    )

if envoyer and question.strip():
    resultats = {}
    for name in CHOIX[methodes or "Les deux"]:
        try:
            with st.spinner(f"{PIPELINES[name]['titre']} : recherche et rédaction…"):
                resultats[name] = ask(name, question, offline)
        except Exception as exc:  # noqa: BLE001 — affiché à l'écran pendant la démo
            resultats[name] = f"{type(exc).__name__} : {exc}"
    st.session_state["resultats"] = {"question": question.strip(), "reponses": resultats}

# Les resultats restent en memoire de session : un clic (sous-graphe...) ne les efface plus.
etat = st.session_state.get("resultats")
if etat:
    st.markdown(
        f'<div class="question-posee">« {html.escape(etat["question"])} »</div>',
        unsafe_allow_html=True,
    )
    reponses = etat["reponses"]
    cols = st.columns(len(reponses), gap="large")
    for col, (name, res) in zip(cols, reponses.items(), strict=True):
        with col:
            colonne(name, res)
    r, g = reponses.get("rag"), reponses.get("graphrag")
    if isinstance(r, Answer) and isinstance(g, Answer):
        st.markdown(verdict(r, g), unsafe_allow_html=True)
