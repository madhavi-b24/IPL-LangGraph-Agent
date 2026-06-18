from langgraph.graph import StateGraph, END
from graph.state import IPLAgentState
from graph.nodes import (
    rewrite_query_node,
    router_node,
    tool_router_node,
    batting_stats_node,
    bowling_stats_node,
    venue_node,
    h2h_node,
    form_node,
    records_node,
    synthesis_node,
)
from classification import classify_query_node
from graph.team_node import team_profile_node
from graph.validation import validation_node
from verification.answer_verifier import answer_verifier_node
from evaluation.prediction_confidence import prediction_confidence_node
from evaluation import evaluation_node
from memory import memory_node, memory_update_node
from safety.hallucination_guard import hallucination_guard_node
from agents import (
    supervisor_node,
    team_agent_node,
    player_agent_node,
    venue_agent_node,
    h2h_agent_node,
    dream11_agent_node,
    prediction_agent_node,
)


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


def route_supervisor(state: IPLAgentState) -> str:
    selected_agent = state.get("selected_agent", "")
    routing_map = {
        "TeamAgent": "team_agent",
        "PlayerAgent": "player_agent",
        "VenueAgent": "venue_agent",
        "H2HAgent": "h2h_agent",
        "Dream11Agent": "dream11_agent",
        "PredictionAgent": "prediction_agent",
    }
    return routing_map.get(selected_agent, "rewrite")


def build_ipl_graph():
    """
    Full IPL LangGraph topology with Query Rewriting:

    ┌──────────────────────────────────────────────────────────────┐
    │                      RewriteNode                              │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────────────┐
    │                         RouterNode                              │
    └──┬──────┬──────┬──────┬──────┬──────┬──────┬───────────────────┘
       │      │      │      │      │      │      │
     team  batting bowling venue  h2h   form  records
       │      │      │      │      │      │      │
       │      │      │      └──→form→batting→bowling
       │      │      │           │
       │      └──────┴───────────┘
       │                         │
       └──────────────→ validation → synthesis → END
    """
    graph = StateGraph(IPLAgentState)

    # ── Register all nodes ────────────────────────────────────────────
    graph.add_node("rewrite",    rewrite_query_node)
    graph.add_node("memory", memory_node)
    graph.add_node("memory_update", memory_update_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("team_agent", team_agent_node)
    graph.add_node("player_agent", player_agent_node)
    graph.add_node("venue_agent", venue_agent_node)
    graph.add_node("h2h_agent", h2h_agent_node)
    graph.add_node("dream11_agent", dream11_agent_node)
    graph.add_node("prediction_agent", prediction_agent_node)
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
    graph.add_node("hallucination_guard", hallucination_guard_node)
    graph.add_node("answer_verifier", answer_verifier_node)
    graph.add_node("evaluation", evaluation_node)

    # ── Entry point ───────────────────────────────────────────────────
    graph.set_entry_point("memory")

    # ── QueryClassifier → Rewrite → Router edge ───────────────────────
    graph.add_node("query_classifier", classify_query_node)
    graph.add_edge("memory", "query_classifier")
    graph.add_edge("query_classifier", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "team_agent": "team_agent",
            "player_agent": "player_agent",
            "venue_agent": "venue_agent",
            "h2h_agent": "h2h_agent",
            "dream11_agent": "dream11_agent",
            "prediction_agent": "prediction_agent",
            "rewrite": "rewrite",
        },
    )
    graph.add_edge("team_agent", "rewrite")
    graph.add_edge("player_agent", "rewrite")
    graph.add_edge("venue_agent", "rewrite")
    graph.add_edge("h2h_agent", "rewrite")
    graph.add_edge("dream11_agent", "rewrite")
    graph.add_edge("prediction_agent", "rewrite")
    graph.add_edge("rewrite", "router")
    graph.add_node("tool_router", tool_router_node)
    graph.add_edge("router", "tool_router")

    # ── Conditional routing from ToolRouterNode ───────────────────────────────────────
    graph.add_conditional_edges(
        "tool_router",
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
    graph.add_edge("batting", "validation")
    graph.add_edge("bowling", "validation")

    # Simple single-node paths → Synthesis
    graph.add_edge("team",    "validation")
    graph.add_edge("records", "validation")

    # ── Synthesis → Validation → AnswerVerifier → END ─────────────────
    graph.add_edge("validation", "synthesis")
    graph.add_edge("synthesis", "hallucination_guard")
    graph.add_edge("hallucination_guard", "answer_verifier")
    graph.add_edge("answer_verifier", "prediction_confidence")
    graph.add_node("prediction_confidence", prediction_confidence_node)
    graph.add_edge("prediction_confidence", "evaluation")
    graph.add_edge("evaluation", "memory_update")
    graph.add_edge("memory_update", END)

    return graph.compile()
