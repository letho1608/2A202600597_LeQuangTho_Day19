"""
LAB 19: GraphRAG - US Electric Vehicle Sector
Entry point: python -m graphrag.pipeline
"""
import os
import sys
import csv
from collections import defaultdict

from .config import get_config
from .loader import load_docs
from .graph import build_rule, build_from_llm, print_stats, visualize, get_context
from .flat_rag import FlatRAG
from .llm import get_llm, answer_question

BENCHMARK = [
    ("Tesla co lien quan den nhung cong ty nao?", "Tesla"),
    ("Cac hang xe dien ban tai My?", "United States"),
    ("McKinsey danh gia thi truong EV ra sao?", "McKinsey"),
    ("Uu diem cua xe dien?", "Battery Electric Vehicle"),
    ("Cox Automotive ve doanh so EV?", "Cox Automotive"),
    ("So sanh EV vs xe xang co loi ich gi?", "ICE"),
    ("Ha tang sac dien o My phat trien the nao?", "Charging Station"),
    ("Vai tro cua California trong nganh EV?", "California"),
    ("Chinh sach nao ho tro phat trien EV?", "EPA"),
    ("Thi truong EV toan cau nam 2024?", "BloombergNEF"),
]


def main():
    print("=" * 60)
    print("LAB 19: GRAPHRAG - US ELECTRIC VEHICLE SECTOR")
    print("=" * 60)

    print("\n[1/5] LOADING DOCUMENTS...")
    docs = load_docs()
    print(f"  Loaded {len(docs)} docs")

    print("\n[2/5] ENTITY & RELATION EXTRACTION...")
    llm = get_llm()
    cfg = get_config()

    if llm:
        print(f"  LLM: provider={cfg['provider']}, model={cfg['model']}")
        G = build_from_llm(docs, llm)
    else:
        print("  Using rule-based extraction")
        G = build_rule(docs)

    print_stats(G)

    print("\n[3/5] RENDERING GRAPH...")
    visualize(G)

    print("\n[4/5] INDEXING FLAT RAG...")
    flat = FlatRAG()
    flat.index(docs)

    print("\n[5/5] EVALUATION (10 benchmark questions)...")
    all_rows = []
    for i, (q, ent) in enumerate(BENCHMARK):
        g_ctx = get_context(G, ent)
        f_res = flat.search(q, k=3)
        f_ctx = "\n".join([r[0]["text"][:300] for r in f_res]) if f_res else "No results"

        if llm and len(g_ctx) > 10:
            g_ans = answer_question(llm, q, g_ctx)
        else:
            g_ans = g_ctx[:200]

        if llm and len(f_ctx) > 10:
            f_ans = answer_question(llm, q, f_ctx)
        else:
            f_ans = f_ctx[:200]

        g_ok = len(g_ans) > 20
        f_ok = len(f_ans) > 20
        all_rows.append({
            "question": q, "entity": ent,
            "graph_ok": g_ok, "flat_ok": f_ok,
            "graph_answer": g_ans[:200].replace("\n", " "),
            "flat_answer": f_ans[:200].replace("\n", " "),
        })
        status = f"[{'G' if g_ok else ' '}{'F' if f_ok else ' '}]"
        print(f"  Q{i+1}: {q:<45s} {status}")

    # Summary table
    print("\n" + "-" * 60)
    print("COMPARISON SUMMARY")
    print("-" * 60)
    print(f"{'#':<3} {'Question':<40} {'GraphRAG':<10} {'FlatRAG':<10}")
    print("-" * 60)
    g_tot = f_tot = 0
    for i, r in enumerate(all_rows):
        g = "CO" if r["graph_ok"] else "THIEU"
        f = "CO" if r["flat_ok"] else "THIEU"
        if r["graph_ok"]: g_tot += 1
        if r["flat_ok"]: f_tot += 1
        print(f"{i+1:<3} {r['question'][:38]:<40} {g:<10} {f:<10}")
    print("-" * 60)
    print(f"Tong:  GraphRAG {g_tot}/10 | FlatRAG {f_tot}/10")

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "comparison_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys(), escapechar="\\", quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Exported: {csv_path}")

    # Cost estimate
    total_tokens = sum(len(d["content"].split()) for d in docs)
    print(f"\nToken usage: ~{total_tokens} tokens")
    print(f"Cost estimate: ~${total_tokens * 0.002 / 1000:.4f} (GPT-4o-mini)")
    print(f"Config: {cfg['provider']} / {cfg['model']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
