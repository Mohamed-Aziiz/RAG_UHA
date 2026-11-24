"""Visualisation pyvis du graphe LightRAG (page HTML autonome)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

PALETTE = {
    "Formation": "#1f77b4",
    "Diplome": "#aec7e8",
    "Composante": "#ff7f0e",
    "Laboratoire": "#ffbb78",
    "Service": "#2ca02c",
    "Personne": "#d62728",
    "Cours": "#9467bd",
    "Lieu": "#8c564b",
    "Procedure": "#e377c2",
    "Date": "#7f7f7f",
    "Organisation": "#bcbd22",
    "Other": "#17becf",
}


def subgraph_around(
    graph: nx.Graph, names: list[str], hops: int = 1, max_nodes: int = 80
) -> nx.Graph:
    seeds = [n for n in names if n in graph]
    nodes = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        frontier = {m for n in frontier for m in graph.neighbors(n)} - nodes
        nodes |= frontier
        if len(nodes) >= max_nodes:
            break
    return graph.subgraph(list(nodes)[:max_nodes]).copy()


def top_subgraph(graph: nx.Graph, max_nodes: int = 80) -> nx.Graph:
    top = [n for n, _ in sorted(graph.degree, key=lambda x: -x[1])[:max_nodes]]
    return graph.subgraph(top).copy()


def to_html(graph: nx.Graph, out: Path, height: str = "600px") -> Path:
    from pyvis.network import Network

    net = Network(
        height=height, width="100%", directed=False, notebook=False, cdn_resources="in_line"
    )
    for n, d in graph.nodes(data=True):
        t = str(d.get("entity_type", "Other")).strip('"')
        net.add_node(
            n,
            label=n[:40],
            title=f"{t}\n{str(d.get('description', ''))[:400]}",
            color=PALETTE.get(t, PALETTE["Other"]),
            size=10 + 2 * graph.degree(n),
        )
    for u, v, d in graph.edges(data=True):
        net.add_edge(
            u, v, title=str(d.get("description", ""))[:300], value=float(d.get("weight", 1) or 1)
        )
    net.toggle_physics(True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(net.generate_html(notebook=False), encoding="utf-8")
    return out
