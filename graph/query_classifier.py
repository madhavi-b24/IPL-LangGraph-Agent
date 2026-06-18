import os
from graph.state import IPLAgentState
from graph.nodes import _get_llm

ALLOWED_LABELS = [
    "player",
    "team",
    "venue",
    "head_to_head",
    "dream11",
    "prediction",
    "captain",
    "stats",
    "general",
]


def _normalize_label(text: str) -> str:
    normalized = text.strip().lower().splitlines()[0].strip()
    if normalized in ALLOWED_LABELS:
        return normalized

    for label in ALLOWED_LABELS:
        if label in normalized:
            return label

    return "general"


def classify_query(state: IPLAgentState) -> IPLAgentState:
    query = state.get("rewritten_query") or state["user_query"]
    llm = _get_llm()
    query_type = "general"

    try:
        if llm is not None:
            prompt = f"""You are an IPL query classifier.

Return ONLY one label from this list EXACTLY as written:
{ALLOWED_LABELS}

Query: {query}

Label:"""
            response = llm.invoke(prompt)
            content = getattr(response, "content", None)
            text = str(content or response or "").strip()
            query_type = _normalize_label(text)
        else:
            query_type = "general"
    except Exception as exc:
        print(f"[QUERY CLASSIFIER] classification failed: {exc}")
        query_type = "general"

    print(f"[QUERY CLASSIFIER] Query: {query}")
    print(f"[QUERY CLASSIFIER] Type: {query_type}")

    return {
        **state,
        "query_type": query_type,
    }
