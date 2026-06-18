from graph.state import IPLAgentState


def venue_agent_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("VenueAgent")
    print("[AGENT]")
    print("VenueAgent activated")
    return {
        **state,
        "agent_metadata_filters": {"section": "venue"},
        "agent_retrieval_strategy": "prioritize_venue_section",
        "nodes_activated": activated,
    }
