from langgraph.graph import StateGraph, END
from graph.state import IPLAgentState
from graph.nodes import (
    router_node,
    batting_stats_node,
    bowling_stats_node,
    venue_node,
    h2h_node,
    form_node,
    records_node,
    synthesis_node,
)
from graph.team_node import team_profile_node
from graph.validation import validation_node


def route_query(state: IPLAgentState) -> str:
    """
    Conditional edge after RouterNode.
    Returns the key of the next node to visit.

    Routing logic:
      team/season  → team
      batting      → batting
      bowling      → bowling
      venue        → venue
      h2h          → h2h
      form         → form
      records      → records
      prediction   → h2h  (triggers full H2H→Venue→Form→Batting→Bowling chain)
      dream11      → form (triggers Form→Batting→Bowling chain)
    """
    qt = state["query_type"]
    routing_map = {
        "team":       "team",
        "season":     "team",
        "batting":    "batting",
        "bowling":    "bowling",
        "records":    "records",
        "venue":      "venue",
        "h2h":        "h2h",
        "form":       "form",
        "prediction": "h2h",
        "dream11":    "form",
    }
    return routing_map.get(qt, "records")


def build_ipl_graph():
    """
    Full IPL LangGraph topology:

    ┌─────────────────────────────────────────────────────────────────┐
    │                         RouterNode                               │
    └──┬──────┬──────┬──────┬──────┬──────┬──────┬────────────────────┘
       │      │      │      │      │      │      │
     team  batting bowling venue  h2h   form  records
       │      │      │      │      │      │      │
       │      │      │      └──→form→batting→bowling
       │      │      │           │
       │      └──────┴───────────┘
       │                         │
       └──────────────→ synthesis → validation → END
    """
    graph = StateGraph(IPLAgentState)

    # ── Register all nodes ────────────────────────────────────────────
    graph.add_node("router",     router_node)
    graph.add_node("team",       team_profile_node)
    graph.add_node("batting",    batting_stats_node)
    graph.add_node("bowling",    bowling_stats_node)
    graph.add_node("venue",      venue_node)
    graph.add_node("h2h",        h2h_node)
    graph.add_node("form",       form_node)
    graph.add_node("records",    records_node)
    graph.add_node("synthesis",  synthesis_node)
    graph.add_node("validation", validation_node)

    # ── Entry point ───────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Conditional routing from RouterNode ───────────────────────────
    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "team":    "team",
            "batting": "batting",
            "bowling": "bowling",
            "records": "records",
            "venue":   "venue",
            "h2h":     "h2h",
            "form":    "form",
        },
    )

    # ── Multi-node paths ──────────────────────────────────────────────
    # Prediction: H2H → Venue → Form → Batting → Bowling → Synthesis
    graph.add_edge("h2h",     "venue")
    graph.add_edge("venue",   "form")
    graph.add_edge("form",    "batting")
    graph.add_edge("batting", "bowling")
    graph.add_edge("bowling", "synthesis")

    # Simple single-node paths → Synthesis
    graph.add_edge("team",    "synthesis")
    graph.add_edge("records", "synthesis")

    # ── Synthesis → Validation → END ─────────────────────────────────
    graph.add_edge("synthesis",  "validation")
    graph.add_edge("validation", END)

    return graph.compile()