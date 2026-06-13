import time
import json
import hashlib
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
    layout="wide",
)

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.stTextInput > div > div > input {
    border-radius: 15px;
}

.answer-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #eef6ff;
    border-left: 5px solid #2563eb;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
### 🧠 Multi-Agent IPL Analysis System

Batting • Bowling • Venue • H2H • Form • Records • Validation
""")
st.info(
"LangGraph → Retrieval → Specialized Agents → Llama 3 → Validation"
)
col1,col2,col3 = st.columns(3)

col1.metric("Agents", "8")
col2.metric("LLM", "Llama 3")
col3.metric("Vector DB", "ChromaDB")
st.caption("Powered by LangGraph • Groq (Llama 3) • ChromaDB • RAG")
st.success("✅ Multi-Agent LangGraph Workflow Active")
# ── Persistent history file (survives page reload like ChatGPT) ───────────────
HISTORY_FILE = "chat_history.json"

def load_persistent_history() -> list:
    """Load history from disk — survives Streamlit page reloads."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_persistent_history(history: list):
    """Save history to disk after every new message."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

# ── Session state init ────────────────────────────────────────────────────────
# chat_history   : full Q&A list shown in main area (loaded from disk on first run)
# query_cache    : {hash -> answer} so same question returns instantly
# memory_context : last 3 Q&A pairs sent to LLM as conversational context
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_persistent_history()
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}
if "memory_context" not in st.session_state:
    # Rebuild memory from loaded history (last 3 turns)
    st.session_state.memory_context = st.session_state.chat_history[-3:]


# ── Load graph (cached so it runs only once) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_graph():
    if not os.path.exists(CHROMA_DIR):
        with st.spinner("⏳ First run — ingesting IPL dataset into vector store..."):
            ingest_pdf("data/IPL_LangGraph_RAG_Dataset.pdf")
    return build_ipl_graph()

graph = load_graph()

# ── Helpers ───────────────────────────────────────────────────────────────────
def query_hash(q: str) -> str:
    return hashlib.md5(q.strip().lower().encode()).hexdigest()

def build_memory_prompt(query: str) -> str:
    """
    Prepend last 3 Q&A turns to current query so the LLM has
    conversational context — e.g. 'compare him with Rohit' after
    asking about Kohli will still make sense.
    """
    memory = st.session_state.memory_context
    if not memory:
        return query
    context_block = "\n".join(
        f"User: {m['query']}\nAssistant: {m['answer']}"
        for m in memory[-3:]
    )
    return (
        f"Previous conversation:\n{context_block}\n\n"
        f"New question: {query}"
    )

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

def run_query(query: str):
    """
    1. Check cache — return instantly if seen before.
    2. Inject conversational memory into query.
    3. Run through LangGraph.
    4. Store result in cache.
    Returns: (answer, time_taken_seconds)
    """
    h = query_hash(query)

   # ── Cache hit ─────────────────────────────────────────────────────────────
    if h in st.session_state.query_cache:
        cached_answer = st.session_state.query_cache[h]
        return cached_answer, 0.04   # near-instant — shows cache effect clearly
    # ── Memory-augmented query ────────────────────────────────────────────────
    memory_query = build_memory_prompt(query)

    # ── LangGraph call ────────────────────────────────────────────────────────
    start = time.time()
    result = graph.invoke(empty_state(memory_query))
    elapsed = round(time.time() - start, 2)
    answer = result["final_answer"]

    # ── Store in cache ────────────────────────────────────────────────────────
    st.session_state.query_cache[h] = answer

    return answer, elapsed

# ── Sidebar: sample queries ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 IPL Assistant")
    st.divider()

    # ── History (persists across reloads) ─────────────────────────────────────
    # Show unique questions only — grouped like ChatGPT sidebar
    # One entry per unique question, most recent first
    seen = set()
    unique_history = []
    for item in reversed(st.session_state.chat_history):
        if item["query"] not in seen:
            seen.add(item["query"])
            unique_history.append(item["query"])

    if unique_history:
        st.markdown("#### 🕘 History")
        for past_q in unique_history[:25]:   # cap at 25 sidebar items
            label = past_q[:38] + "..." if len(past_q) > 38 else past_q
            if st.button(label, key=f"hist_{query_hash(past_q)}", use_container_width=True):
                st.session_state["selected_query"] = past_q

        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory_context = []
            st.session_state.query_cache = {}
            save_persistent_history([])
            st.rerun()

        st.divider()
# ── Suggested queries ──────────────────────────────────────────────────────
    st.markdown("#### 💡 Try these queries")
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
        if st.button(sq, key=f"sug_{query_hash(sq)}", use_container_width=True):
            st.session_state["selected_query"] = sq

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Multi-agent RAG using **LangGraph**. "
        "Queries are routed to specialised nodes "
        "(Batting, Bowling, Venue, H2H, Form) "
        "and synthesised by **Groq Llama 3.1**."
    )


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.title("🏏 IPL Intelligence Assistant")
st.caption("Powered by LangGraph · Groq (Llama 3.1) · ChromaDB · RAG")

# ── Show full conversation history ────────────────────────────────────────────
if st.session_state.chat_history:
    for item in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.markdown(item["query"])

        # Assistant message
        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            # Response time shown below each answer
            t = item["time_taken"]
            if t < 0.5:
                st.caption(f"⚡ Response time: **{t}s** (cached)")
            else:
                st.caption(f"🕐 Response time: **{t}s**")

# ── Input bar ─────────────────────────────────────────────────────────────────
default_q = st.session_state.pop("selected_query", "")
query = st.chat_input(
    placeholder="Ask anything about IPL — stats, predictions, Dream11...",
)

# Allow sidebar history / suggested query buttons to pre-fill
if not query and default_q:
    query = default_q

# ── Handle query ──────────────────────────────────────────────────────────────
if query:
    query = query.strip()

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Run query (with cache + memory)
    with st.chat_message("assistant"):
        with st.spinner("Agents thinking..."):
            answer, elapsed = run_query(query)

        st.markdown(answer)

        if elapsed < 0.5:
            st.caption(f"⚡ Response time: **{elapsed}s** (cached)")
        else:
            st.caption(f"🕐 Response time: **{elapsed}s**")

    # ── Save to history ───────────────────────────────────────────────────────
    entry = {"query": query, "answer": answer, "time_taken": elapsed}
    st.session_state.chat_history.append(entry)
    save_persistent_history(st.session_state.chat_history)

    # ── Update conversational memory (keep last 3 turns) ──────────────────────
    st.session_state.memory_context.append(entry)
    st.session_state.memory_context = st.session_state.memory_context[-3:]

    st.rerun()