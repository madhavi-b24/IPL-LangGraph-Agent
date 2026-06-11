import os
from langchain_groq import ChatGroq
from graph.state import IPLAgentState
from rag.retriever import retrieve

# ── LLM (shared across all nodes) ──────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)


# ── NODE 1: RouterNode ──────────────────────────────────────────────────────
def router_node(state: IPLAgentState) -> IPLAgentState:
    """Classifies query type and extracts team/player entity names."""
    query = state["user_query"].lower()

    # Entity extraction
    teams = ["mi", "csk", "rcb", "kkr", "dc", "srh", "rr", "pbks", "lsg", "gt"]
    players = [
        "kohli", "rohit", "bumrah", "dhoni", "buttler", "head",
        "rashid", "gaikwad", "samson", "rahul", "pandya", "narine",
        "chahal", "gill", "pant", "klaasen", "sharma", "warner"
    ]
    entities = []
    for t in teams:
        if t in query:
            entities.append(t.upper())
    for p in players:
        if p in query:
            entities.append(p.capitalize())

    # Query type routing
    if any(w in query for w in ["dream11", "fantasy", "captain pick", "xi", "team pick"]):
        query_type = "dream11"
    elif any(w in query for w in ["predict", "win", "who will", "likely to win", "winner"]):
        query_type = "prediction"
    elif any(w in query for w in ["record", "highest", "most", "fastest", "best figures", "milestone"]):
        query_type = "records"
    elif any(w in query for w in ["venue", "pitch", "stadium", "ground", "dew", "strategy at"]):
        query_type = "venue"
    elif any(w in query for w in ["form", "last 5", "recent", "this season", "current form"]):
        query_type = "form"
    elif any(w in query for w in ["bowl", "wicket", "economy", "spell", "figures"]):
        query_type = "bowling"
    elif any(w in query for w in ["bat", "run", "average", "century", "strike rate", "opener", "score"]):
        query_type = "batting"
    elif any(w in query for w in ["vs", "head-to-head", "h2h", "played each other", "between"]):
        query_type = "h2h"
    elif any(w in query for w in ["season", "2019", "2020", "2021", "2022", "2023", "2024", "consistent"]):
        query_type = "season"
    elif any(w in query for w in ["team", "captain", "coach", "title", "squad"]):
        query_type = "team"
    else:
        query_type = "records"  # default fallback

    activated = state.get("nodes_activated", [])
    activated.append("RouterNode")

    return {
        **state,
        "query_type": query_type,
        "entities": entities,
        "nodes_activated": activated,
    }


# ── NODE 2: BattingStatsNode ────────────────────────────────────────────────
def batting_stats_node(state: IPLAgentState) -> IPLAgentState:
    docs = retrieve(state["user_query"], section="batting")
    activated = state.get("nodes_activated", [])
    activated.append("BattingStatsNode")
    return {**state, "batting_context": docs, "nodes_activated": activated}


# ── NODE 3: BowlingStatsNode ────────────────────────────────────────────────
def bowling_stats_node(state: IPLAgentState) -> IPLAgentState:
    docs = retrieve(state["user_query"], section="bowling")
    activated = state.get("nodes_activated", [])
    activated.append("BowlingStatsNode")
    return {**state, "bowling_context": docs, "nodes_activated": activated}


# ── NODE 4: VenueNode ───────────────────────────────────────────────────────
def venue_node(state: IPLAgentState) -> IPLAgentState:
    docs = retrieve(state["user_query"], section="venue")
    activated = state.get("nodes_activated", [])
    activated.append("VenueNode")
    return {**state, "venue_context": docs, "nodes_activated": activated}


# ── NODE 5: H2HNode ─────────────────────────────────────────────────────────
def h2h_node(state: IPLAgentState) -> IPLAgentState:
    docs = retrieve(state["user_query"], section="h2h")
    activated = state.get("nodes_activated", [])
    activated.append("H2HNode")
    return {**state, "h2h_context": docs, "nodes_activated": activated}


# ── NODE 6: FormNode ────────────────────────────────────────────────────────
def form_node(state: IPLAgentState) -> IPLAgentState:
    docs = retrieve(state["user_query"], section="form")
    activated = state.get("nodes_activated", [])
    activated.append("FormNode")
    return {**state, "form_context": docs, "nodes_activated": activated}


# ── NODE 7: RecordsNode ─────────────────────────────────────────────────────
def records_node(state: IPLAgentState) -> IPLAgentState:
    docs = retrieve(state["user_query"], section="records")
    activated = state.get("nodes_activated", [])
    activated.append("RecordsNode")
    return {**state, "retrieved_chunks": docs, "nodes_activated": activated}


# ── NODE 8: SynthesisNode ───────────────────────────────────────────────────
def synthesis_node(state: IPLAgentState) -> IPLAgentState:
    """Combines all retrieved context and calls LLM for final answer."""
    all_docs = (
        state.get("batting_context", [])
        + state.get("bowling_context", [])
        + state.get("h2h_context", [])
        + state.get("venue_context", [])
        + state.get("form_context", [])
        + state.get("retrieved_chunks", [])
    )

    activated = state.get("nodes_activated", [])
    activated.append("SynthesisNode")

    if not all_docs:
        return {
            **state,
            "final_answer": (
                "I don't have enough information in my dataset to answer this. "
                "This might be out of scope or not covered in the IPL dataset."
            ),
            "sources": [],
            "nodes_activated": activated,
        }

    context_text = "\n\n".join(d.page_content for d in all_docs[:10])  # cap at 10 chunks
    sources = list(set(
        d.metadata.get("source", "IPL Dataset") for d in all_docs
    ))

    prompt = f"""You are an expert IPL cricket analyst assistant.
Use ONLY the context below to answer the query. Be specific and cite numbers where available.
If the answer is not in the context, clearly say: "This information is not available in my dataset."

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

Query: {state["user_query"]}

Answer (be concise and factual):"""

    response = llm.invoke(prompt)

    return {
        **state,
        "final_answer": response.content,
        "sources": sources,
        "nodes_activated": activated,
    }