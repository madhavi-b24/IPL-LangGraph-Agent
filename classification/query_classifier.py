import json
import os
from difflib import SequenceMatcher
from typing import Dict

from langchain_groq import ChatGroq
from graph.state import IPLAgentState
from rag.retriever import detect_entities


_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            _llm = None
        else:
            _llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.0)
    return _llm


CLASS_EXAMPLES = {
    "team_query": "What is CSK captain?",
    "player_query": "Virat Kohli strike rate",
    "venue_query": "Chinnaswamy pitch report",
    "h2h_query": "MI vs CSK head to head",
    "dream11_query": "Best Dream11 team for CSK vs RCB",
    "prediction_query": "Who will win MI vs CSK",
    "records_query": "Most sixes in IPL",
    "general_query": "What is IPL?",
}


def _semantic_fallback(query: str) -> Dict[str, object]:
    # Compute simple lexical similarity to examples (semantic proxy)
    best_type = "general_query"
    best_score = 0.0
    for t, example in CLASS_EXAMPLES.items():
        score = SequenceMatcher(None, query.lower(), example.lower()).ratio()
        if score > best_score:
            best_score = score
            best_type = t

    # Slight boost if entity detection suggests teams/players
    entities = detect_entities(query)
    if entities.get("teams") and len(entities.get("teams")) >= 2:
        best_type = "h2h_query"
        best_score = max(best_score, 0.75)
    elif entities.get("teams") and len(entities.get("teams")) == 1:
        best_type = "team_query"
        best_score = max(best_score, 0.65)
    elif entities.get("players"):
        best_type = "player_query"
        best_score = max(best_score, 0.65)

    return {"query_type": best_type, "confidence": float(best_score)}


def _extract_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        # naive extraction
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return None
    return None


def classify_query_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("QueryClassifier")

    raw_query = state.get("user_query", "")

    try:
        llm = _get_llm()
        result = None

        if llm is not None:
            prompt = f"""
You are a query classification assistant for an IPL QA system.

Classify the user's query into one of these classes:
team_query, player_query, venue_query, h2h_query, dream11_query, prediction_query, records_query, general_query

Return JSON only: {{"query_type": "...", "confidence": 0.0}}

Query: {raw_query}
"""
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", "") or str(resp)
            parsed = _extract_json(content)
            if parsed and isinstance(parsed, dict) and parsed.get("query_type"):
                qtype = parsed.get("query_type")
                conf = float(parsed.get("confidence", 0.0))
                result = {"query_type": qtype, "confidence": conf}

        if not result:
            # semantic fallback
            result = _semantic_fallback(raw_query)

        qtype = result.get("query_type", "general_query")
        conf = float(result.get("confidence", 0.0))

        # Enforce low-confidence policy
        if conf < 0.50:
            qtype = "general_query"
            conf = 0.0

        print("[QUERY CLASSIFIER]")
        print(f"Query: {raw_query}")
        print(f"Type: {qtype}")
        print(f"Confidence: {conf:.2f}")

        return {
            **state,
            "query_type": qtype,
            "query_confidence": conf,
            "nodes_activated": activated,
        }

    except Exception as exc:
        print("[QUERY CLASSIFIER] Failed:", exc)
        return {
            **state,
            "query_type": "general_query",
            "query_confidence": 0.0,
            "nodes_activated": activated,
        }
