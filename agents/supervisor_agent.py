from graph.state import IPLAgentState
from rag.retriever import detect_entities, flatten_entities


AGENT_RULES = {
    "team_query": ("TeamAgent", 0.91),
    "player_query": ("PlayerAgent", 0.91),
    "venue_query": ("VenueAgent", 0.91),
    "h2h_query": ("H2HAgent", 0.91),
    "dream11_query": ("Dream11Agent", 0.91),
    "prediction_query": ("PredictionAgent", 0.91),
    "records_query": ("TeamAgent", 0.82),
}

TOOL_AGENT_RULES = {
    "team_tool": ("TeamAgent", 0.86),
    "player_tool": ("PlayerAgent", 0.86),
    "venue_tool": ("VenueAgent", 0.86),
    "h2h_tool": ("H2HAgent", 0.86),
    "dream11_tool": ("Dream11Agent", 0.86),
    "prediction_tool": ("PredictionAgent", 0.86),
}


def _agent_from_query_terms(query: str, entities: list[str]) -> tuple[str, float]:
    lowered = query.lower()
    if any(term in lowered for term in ["dream11", "fantasy", "captain pick", "vice captain", " xi "]):
        return "Dream11Agent", 0.91

    prediction_terms = ["predict", "who will win", "likely to win", "winner", "win probability", "favoured", "favored"]
    if any(term in lowered for term in prediction_terms) and len(entities) >= 2:
        return "PredictionAgent", 0.91

    return "", 0.0


def supervisor_node(state: IPLAgentState) -> IPLAgentState:
    query_type = state.get("query_type", "")
    selected_tool = state.get("selected_tool")
    entities = state.get("entities", [])
    if not entities:
        entities = flatten_entities(detect_entities(state.get("user_query", "")))

    selected_agent, confidence = _agent_from_query_terms(state.get("user_query", ""), entities)
    if not selected_agent:
        selected_agent, confidence = AGENT_RULES.get(query_type, ("", 0.0))
    if not selected_agent and selected_tool:
        selected_agent, confidence = TOOL_AGENT_RULES.get(selected_tool, ("", 0.0))

    activated = state.get("nodes_activated", [])
    activated.append("Supervisor")

    print("[SUPERVISOR]")
    print(f"Selected Agent: {selected_agent or 'ExistingRetrievalPath'}")
    print(f"Confidence: {confidence:.2f}")

    return {
        **state,
        "selected_agent": selected_agent,
        "agent_confidence": confidence,
        "nodes_activated": activated,
    }
