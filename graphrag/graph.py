import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import List, Dict, Tuple

from .entities import ALL_ENTS, ENTITIES, extract_entities, extract_relations


def build_rule(docs: List[Dict]) -> nx.MultiDiGraph:
    entity_doc_map = defaultdict(set)
    all_rels = []
    for doc in docs:
        ents = extract_entities(doc["content"])
        for en, ec in ents:
            entity_doc_map[en].add(doc["id"])
        all_rels.extend(extract_relations(doc["content"], ents))

    G = nx.MultiDiGraph()
    for en, ec in ALL_ENTS.items():
        if en in entity_doc_map:
            G.add_node(en, category=ec, doc_ids=list(entity_doc_map[en]))
    for s, r, o in list(set(all_rels)):
        if G.has_node(s) and G.has_node(o):
            G.add_edge(s, o, relation=r)
    return G


def build_from_llm(docs: List[Dict], llm) -> nx.MultiDiGraph:
    from .llm import extract_triples_llm
    print("  Extracting triples via LLM (batch)...")
    triples = extract_triples_llm(llm, docs, batch_size=3)
    print(f"  Extracted {len(triples)} triples")

    G = nx.MultiDiGraph()
    for t in triples:
        h, r, tl = t.get("head", ""), t.get("relation", ""), t.get("tail", "")
        if h and tl:
            if not G.has_node(h):
                G.add_node(h, category="EXTRACTED")
            if not G.has_node(tl):
                G.add_node(tl, category="EXTRACTED")
            G.add_edge(h, tl, relation=r)
    return G


def print_stats(G: nx.MultiDiGraph):
    print(f"\n  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    ec = defaultdict(int)
    for _, d in G.nodes(data=True):
        ec[d.get("category", "?")] += 1
    for k, v in sorted(ec.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")
    print("\n  Top entities (by degree):")
    for name, deg in sorted(dict(G.degree()).items(), key=lambda x: -x[1])[:15]:
        cat = G.nodes[name].get("category", "")
        print(f"    {name:<35s} [{cat:<15s}] deg={deg}")


def visualize(G: nx.MultiDiGraph, output: str = "knowledge_graph.png"):
    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(G, k=2, iterations=30)
    cmap = {
        "COMPANY": "#FF6B6B", "ORGANIZATION": "#4ECDC4", "PERSON": "#FFE66D",
        "LOCATION": "#95E1D3", "TECHNOLOGY": "#F38181", "METRIC": "#AA96DA",
        "EXTRACTED": "#B8B8B8",
    }
    colors = [cmap.get(G.nodes[n].get("category", ""), "#B8B8B8") for n in G.nodes()]
    sizes = [max(600, min(2500, G.degree(n) * 200)) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=True, arrowstyle="-|>", arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold")
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=k)
        for k, c in cmap.items()
    ]
    plt.legend(handles=legend, loc="upper left", fontsize=11)
    plt.title("Knowledge Graph - US Electric Vehicle Sector", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output}")


def get_context(G: nx.MultiDiGraph, entity: str) -> str:
    if not G.has_node(entity):
        return f"Entity '{entity}' not in graph"
    ctx = []
    for u, v, k, d in G.edges(entity, keys=True, data=True):
        ctx.append(f"({entity})--[{d.get('relation', '')}]-->({v})")
    for u, v, k, d in G.in_edges(entity, keys=True, data=True):
        ctx.append(f"({u})--[{d.get('relation', '')}]-->({entity})")
    one_hop = set()
    for u, v, k in G.edges(entity, keys=True):
        one_hop.add(v)
    for u, v, k in G.in_edges(entity, keys=True):
        one_hop.add(u)
    for hop in one_hop:
        for u, v, k, d in G.edges(hop, keys=True, data=True):
            if v != entity and v not in one_hop:
                ctx.append(f"({hop})--[{d.get('relation', '')}]-->({v}) [2-hop]")
        for u, v, k, d in G.in_edges(hop, keys=True, data=True):
            if u != entity and u not in one_hop:
                ctx.append(f"({u})--[{d.get('relation', '')}]-->({hop}) [2-hop]")
    return "\n".join(ctx[:20]) if ctx else "No context"
