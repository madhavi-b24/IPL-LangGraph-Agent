import os
from dotenv import load_dotenv
load_dotenv()

from rag.ingest import ingest_pdf, CHROMA_DIR
from graph.graph_builder import build_ipl_graph

# ── Step 1: Ingest PDF (only if vector store doesn't exist yet) ─────────────
if not os.path.exists(CHROMA_DIR):
    ingest_pdf("data/IPL_LangGraph_RAG_Dataset.pdf")
else:
    print("Vector store already exists — skipping ingestion.")

# ── Step 2: Build graph ──────────────────────────────────────────────────────
print("\nBuilding LangGraph...\n")
graph = build_ipl_graph()

# ── Step 3: Default empty state template ────────────────────────────────────
def empty_state(query: str) -> dict:
    return {
        "user_query": query,
        "query_type": "",
        "entities": [],
        "batting_context": [],
        "bowling_context": [],
        "h2h_context": [],
        "venue_context": [],
        "form_context": [],
        "retrieved_chunks": [],
        "synthesised_context": "",
        "final_answer": "",
        "sources": [],
        "confidence": 0.0,
        "nodes_activated": [],
    }

# ── Step 4: Test queries (maps to Evaluation Set in Section 12) ──────────────
test_queries = [
    # Easy
    "Who captains Chennai Super Kings in 2024?",
    "What is Virat Kohli's career IPL run tally?",
    "What is the highest team total in IPL history?",
    # Medium
    "List all bowlers with an economy rate below 7.0.",
    "Which opener has the highest strike rate?",
    "How many times have MI and CSK played each other?",
    # Hard
    "Suggest a Dream11 XI for MI vs SRH at Wankhede tonight.",
    "CSK is playing RCB at Chinnaswamy. Who will win?",
]

for query in test_queries:
    print("=" * 60)
    print(f"Q: {query}")
    result = graph.invoke(empty_state(query))
    print(f"Route  : {result['query_type']}")
    print(f"Nodes  : {' → '.join(result['nodes_activated'])}")
    print(f"Answer : {result['final_answer'][:300]}")
    print()