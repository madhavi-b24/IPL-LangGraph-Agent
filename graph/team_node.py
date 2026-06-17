"""
TeamProfileNode — entry-point node for team identity queries.
Handles: captain, coach, home ground, titles, playing style.
If the query also needs player stats, it passes entities downstream
to BattingStatsNode or BowlingStatsNode automatically.
"""

from graph.state import IPLAgentState
from rag.retriever import retrieve
from graph.nodes import _resolve_retrieval_query


def team_profile_node(state: IPLAgentState) -> IPLAgentState:
    """
    Retrieves team profile data.
    Tagged section: 'team' in ChromaDB.
    """
    query = _resolve_retrieval_query(state)
    if state.get("query_type") == "season":
        all_docs = retrieve(query, section="season", k=4)
    else:
        all_docs = retrieve(query, section="team", k=6)

    activated = state.get("nodes_activated", [])
    activated.append("TeamProfileNode")
    print(f"[TeamProfileNode] query='{query}' retrieved={len(all_docs)} docs")

    # Store in retrieved_chunks so SynthesisNode picks it up
    return {
        **state,
        "retrieved_chunks": all_docs,
        "nodes_activated": activated,
    }
