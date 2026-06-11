"""
TeamProfileNode — entry-point node for team identity queries.
Handles: captain, coach, home ground, titles, playing style.
If the query also needs player stats, it passes entities downstream
to BattingStatsNode or BowlingStatsNode automatically.
"""

from graph.state import IPLAgentState
from rag.retriever import retrieve


def team_profile_node(state: IPLAgentState) -> IPLAgentState:
    """
    Retrieves team profile data.
    Tagged section: 'team' in ChromaDB.
    """
    docs = retrieve(state["user_query"], section="team", k=4)

    # Also pull season data for richer context on trend queries
    season_docs = retrieve(state["user_query"], section="season", k=3)
    all_docs = docs + season_docs

    activated = state.get("nodes_activated", [])
    activated.append("TeamProfileNode")

    # Store in retrieved_chunks so SynthesisNode picks it up
    return {
        **state,
        "retrieved_chunks": all_docs,
        "nodes_activated": activated,
    }