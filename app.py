import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from rag.ingest import ingest_pdf, CHROMA_DIR
from graph.graph_builder import build_ipl_graph

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Intelligence Assistant",
    page_icon="🏏",
    layout="centered",
)

st.title("🏏 IPL Intelligence Assistant")
st.caption("Powered by LangGraph · Groq (Llama 3) · ChromaDB · RAG")

# ── Load graph (cached so it runs only once) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_graph():
    if not os.path.exists(CHROMA_DIR):
        with st.spinner("⏳ First run — ingesting IPL dataset into vector store..."):
            ingest_pdf("data/IPL_LangGraph_RAG_Dataset.pdf")
    return build_ipl_graph()

graph = load_graph()

# ── Empty state helper ────────────────────────────────────────────────────────
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

# ── Sidebar: sample queries ───────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Try these queries")
    sample_queries = [
        "Who captains Chennai Super Kings in 2024?",
        "What is Virat Kohli's career IPL run tally?",
        "List all bowlers with an economy rate below 7.0.",
        "Which opener has the highest strike rate?",
        "How many times have MI and CSK played each other?",
        "Suggest a Dream11 XI for MI vs SRH at Wankhede tonight.",
        "CSK is playing RCB at Chinnaswamy. Who will win?",
        "What bowling strategy should SRH use at MA Chidambaram against CSK?",
    ]
    for sq in sample_queries:
        if st.button(sq, use_container_width=True):
            st.session_state["selected_query"] = sq

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This app uses a **LangGraph multi-agent RAG** system. "
        "Each query is routed to specialised nodes (Batting, Bowling, Venue, H2H, Form) "
        "before being synthesised by an LLM."
    )

# ── Main input ────────────────────────────────────────────────────────────────
default_q = st.session_state.get("selected_query", "")
query = st.text_input(
    "Ask anything about IPL:",
    value=default_q,
    placeholder="e.g. Suggest a Dream11 XI for MI vs SRH at Wankhede tonight",
)

if query:
    with st.spinner("🤖 Agents thinking..."):
        result = graph.invoke(empty_state(query))

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown("### 📋 Answer")
    st.write(result["final_answer"])

    # ── Graph trace ───────────────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔀 Query routed as**")
        st.code(result["query_type"])

    with col2:
        st.markdown("**🏷️ Entities detected**")
        entities = result.get("entities", [])
        st.code(", ".join(entities) if entities else "None")

    st.markdown("**🧩 Nodes activated (in order)**")
    nodes = result.get("nodes_activated", [])
    node_chain = " → ".join(nodes)
    st.code(node_chain)

    # ── Show retrieved context (optional expander) ────────────────────────────
    all_docs = (
        result.get("batting_context", [])
        + result.get("bowling_context", [])
        + result.get("h2h_context", [])
        + result.get("venue_context", [])
        + result.get("form_context", [])
        + result.get("retrieved_chunks", [])
    )
    if all_docs:
        with st.expander("📄 Retrieved context chunks"):
            for i, doc in enumerate(all_docs[:6], 1):
                section = doc.metadata.get("section", "?")
                st.markdown(f"**Chunk {i}** · `section={section}`")
                st.text(doc.page_content[:300])
                st.divider()