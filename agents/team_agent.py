from graph.state import IPLAgentState


def team_agent_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("TeamAgent")
    print("[AGENT]")
    print("TeamAgent activated")
    return {
        **state,
        "agent_metadata_filters": {"section": "team"},
        "agent_retrieval_strategy": "prioritize_team_section",
        "nodes_activated": activated,
    }
