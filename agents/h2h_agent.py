from graph.state import IPLAgentState


def h2h_agent_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("H2HAgent")
    print("[AGENT]")
    print("H2HAgent activated")
    return {
        **state,
        "agent_metadata_filters": {"section": "h2h"},
        "agent_retrieval_strategy": "prioritize_h2h_section",
        "nodes_activated": activated,
    }
