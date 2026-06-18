from graph.state import IPLAgentState


def dream11_agent_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("Dream11Agent")
    print("[AGENT]")
    print("Dream11Agent activated")
    return {
        **state,
        "agent_metadata_filters": {"section": "dream11"},
        "agent_retrieval_strategy": "prioritize_dream11_section",
        "nodes_activated": activated,
    }
