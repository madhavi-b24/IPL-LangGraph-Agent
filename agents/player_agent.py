from graph.state import IPLAgentState


def player_agent_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("PlayerAgent")
    print("[AGENT]")
    print("PlayerAgent activated")
    return {
        **state,
        "agent_metadata_filters": {"section": ["batting", "bowling", "player"]},
        "agent_retrieval_strategy": "prioritize_batting_bowling_player_sections",
        "nodes_activated": activated,
    }
