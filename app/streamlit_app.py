"""Interface de démo (une page) : question → RAG, GraphRAG ou les deux côte à côte.

    uv run streamlit run app/streamlit_app.py

Mode hors ligne : si l'API est indisponible, les réponses enregistrées dans
`data/demo_cache.json` (créé par scripts/record_demo.py) sont affichées à la place.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_uha.config import load_settings
from rag_uha.types import Answer

settings = load_settings()
_icon = settings.app.assistant_icon_path
st.set_page_config(
    page_title=settings.app.name,
    page_icon=str(_icon) if _icon.exists() else "🎓",
    layout="wide",
)


@st.cache_resource(show_spinner="Chargement du pipeline…")
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


def _data_uri(path: Path) -> str:
    """Image locale encodée pour l'insérer dans du HTML (st.markdown ne sert pas les fichiers)."""
    mime = "image/svg+xml" if path.suffix == ".svg" else f"image/{path.suffix.lstrip('.')}"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def ask(name: str, question: str, offline: bool) -> Answer:
    if offline:
        cached = demo_cache().get(name, {}).get(question.strip())
        if cached:
            return Answer.from_dict(cached)
        raise RuntimeError("Pas de réponse enregistrée pour cette question en mode hors ligne.")
    return get_pipeline(name).answer(question)


def show_answer(ans: Answer) -> None:
    st.markdown(ans.text)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Latence",
        f"{ans.latency_s:.1f} s",
        help=f"dont recherche : {ans.retrieval_latency_s:.1f} s",
    )
    c2.metric("Tokens LLM", f"{ans.usage.total_tokens}", help=f"{ans.usage.n_calls} appel(s)")
    c3.metric("Sources", f"{len(ans.sources)}")
    with st.expander(f"Sources utilisées ({len(ans.contexts)} passages)"):
        for rc in ans.contexts:
            st.markdown(f"**#{rc.rank} · {rc.chunk.doc_id}** `{rc.chunk.id}` · {rc.chunk.lang}")
            st.caption(rc.chunk.text[:500] + ("…" if len(rc.chunk.text) > 500 else ""))
    if ans.pipeline == "graphrag":
        ents = ans.extra.get("entities", [])
        rels = ans.extra.get("relations", [])
        with st.expander(
            f"Graphe : {len(ents)} entités, {len(rels)} relations (mode {ans.extra.get('mode')})"
        ):
            kw = ans.extra.get("keywords", {})
            if kw:
                st.caption(
                    f"Mots-clés : {', '.join(kw.get('high_level', []) + kw.get('low_level', []))}"
                )
            st.write(", ".join(f"{e['name']} *({e['type']})*" for e in ents[:25]))
            if ents and st.button("Afficher le sous-graphe", key=f"sg-{ans.question[:30]}"):
                from rag_uha.viz import subgraph_around, to_html

                sg = subgraph_around(graph(), [e["name"] for e in ents[:10]], hops=1)
                html = to_html(sg, ROOT / "data" / "subgraph.html")
                st.components.v1.html(html.read_text(encoding="utf-8"), height=520, scrolling=True)


# ------------------------------------------------------------------ page
with st.sidebar:
    sidebar_logos = [
        f'<img src="{_data_uri(p)}" alt="{alt}" style="width:{w};height:auto;min-width:0">'
        for p, alt, w in [
            (settings.app.assistant_logo_path, settings.app.name, "52%"),
            (settings.app.logo_path, "Université de Haute-Alsace", "40%"),
        ]
        if p.exists()
    ]
    if sidebar_logos:
        # Logo de l'assistant | séparateur vertical | logo de l'université.
        sep = '<div style="flex:0 0 2px;align-self:stretch;background:#D9E2EC"></div>'
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            f'gap:10px;margin-bottom:8px">{sep.join(sidebar_logos)}</div>',
            unsafe_allow_html=True,
        )
    st.caption("Assistant documentaire de l'Université de Haute-Alsace — RAG classique vs GraphRAG")
    mode = st.radio(
        "Architecture",
        ["RAG", "GraphRAG", "Les deux"],
        index={"rag": 0, "graphrag": 1, "both": 2}[settings.app.default_mode],
    )
    offline = st.toggle("Mode hors ligne (réponses enregistrées)", value=False)
    st.divider()
    st.caption(
        f"Génération : `{settings.llm_query.model}`  \nEmbeddings : `{settings.embeddings.model}`  \nGraphRAG : mode `{settings.graphrag.mode}`"
    )
    if offline and demo_cache():
        st.caption("Questions enregistrées :")
        for q in list(next(iter(demo_cache().values()), {}).keys())[:8]:
            st.caption(f"• {q}")

if settings.app.assistant_logo_path.exists():
    # Grand logo de l'assistant en tête de page, puis une ligne horizontale.
    st.markdown(
        f'<img src="{_data_uri(settings.app.assistant_logo_path)}" alt="{settings.app.name}"'
        ' style="height:84px">'
        '<hr style="border:none;border-top:2px solid #D9E2EC;margin:20px 0 28px">',
        unsafe_allow_html=True,
    )
else:
    st.header(settings.app.name)
question = st.text_input(
    "Posez votre question (français ou anglais)",
    placeholder="Ex. : Quelles sont les conditions d'admission en Master Data Science ?",
)
go = st.button("Rechercher", type="primary", disabled=not question.strip())

if go and question.strip():
    names = {"RAG": ["rag"], "GraphRAG": ["graphrag"], "Les deux": ["rag", "graphrag"]}[mode]
    cols = st.columns(len(names))
    for col, name in zip(cols, names, strict=True):
        with col:
            st.subheader("RAG classique" if name == "rag" else "GraphRAG")
            try:
                with st.spinner("Recherche et génération…"):
                    ans = ask(name, question, offline)
                show_answer(ans)
            except Exception as exc:  # noqa: BLE001 — affiché à l'écran pendant la démo
                st.error(f"{type(exc).__name__} : {exc}")
                if not offline and demo_cache():
                    st.info("Active le mode hors ligne pour afficher les réponses enregistrées.")
