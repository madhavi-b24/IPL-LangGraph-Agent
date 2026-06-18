from graph.state import IPLAgentState


def prediction_agent_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("PredictionAgent")
    print("[AGENT]")
    print("PredictionAgent activated")
    return {
        **state,
        "agent_metadata_filters": {"section": ["form", "venue", "h2h", "team"]},
        "agent_retrieval_strategy": "prioritize_form_venue_h2h_team_sections",
        "nodes_activated": activated,
    }
