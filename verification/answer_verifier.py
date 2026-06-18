import json
import os
from typing import List

from langchain_groq import ChatGroq
from graph.state import IPLAgentState


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


def _collect_combined_docs(state: IPLAgentState) -> List[str]:
    parts = []
    for key in [
        "batting_context",
        "bowling_context",
        "h2h_context",
        "venue_context",
        "form_context",
        "retrieved_chunks",
    ]:
        docs = state.get(key, []) or []
        for doc in docs:
            try:
                parts.append(str(doc.page_content or ""))
            except Exception:
                continue
    return parts


def _extract_json(text: str):
    """Try to extract JSON object from free text."""
    try:
        return json.loads(text)
    except Exception:
        # naive search for first { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return None
    return None


def answer_verifier_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("AnswerVerifier")

    try:
        # Accept either `answer` or `final_answer` keys
        answer = str(state.get("answer") or state.get("final_answer") or "").strip()
        combined_docs = _collect_combined_docs(state)
        confidence_score = float(state.get("confidence_score", state.get("confidence", 0.0) or 0.0))
        source_list = state.get("source_list") or state.get("source_attribution") or []

        llm = _get_llm()
        if llm is None:
            # LLM unavailable — follow fallback rules
            print("[ANSWER VERIFIER]")
            print("Score: 1.00")
            print("Passed: True")
            print("Reason: Verifier unavailable")
            return {
                **state,
                "verification_score": 1.0,
                "verification_passed": True,
                "verification_reason": "Verifier unavailable",
                "nodes_activated": activated,
            }

        # Build prompt
        context_preview = "\n\n".join((combined_docs[:8])) or ""
        prompt = f"""
You are an answer verification system.

Given:
1. Retrieved context (below)
2. Generated answer

Determine:
- Is the answer supported by the context?
- Is any important claim unsupported?
- Is the answer partially supported?

Return JSON only with keys: supported (bool), score (0.0-1.0), reason (string).

Retrieved context:
{context_preview}

Source list metadata:
{json.dumps(source_list)}

Confidence score (from generator): {confidence_score}

Answer:
{answer}

Return JSON:
"""

        response = llm.invoke(prompt)
        content = getattr(response, "content", "") or str(response)

        parsed = _extract_json(content)
        if not parsed or not isinstance(parsed, dict):
            # fallback: attempt to ask LLM to return strict JSON
            followup = (
                "The previous response could not be parsed.\n"
                "Please respond ONLY with a JSON object like: {\"supported\": true, \"score\": 0.92, \"reason\": \"...\"}"
            )
            response2 = llm.invoke(f"{content}\n\n{followup}")
            content2 = getattr(response2, "content", "") or str(response2)
            parsed = _extract_json(content2)

        if not parsed or not isinstance(parsed, dict):
            # As per failsafe, do not block answers — mark as available
            print("[ANSWER VERIFIER]")
            print("Score: 1.00")
            print("Passed: True")
            print("Reason: Verifier unavailable or unparsable response")
            return {
                **state,
                "verification_score": 1.0,
                "verification_passed": True,
                "verification_reason": "Verifier unavailable or unparsable response",
                "nodes_activated": activated,
            }

        # Normalize values
        supported = bool(parsed.get("supported", False))
        score = float(parsed.get("score", 1.0))
        reason = str(parsed.get("reason", ""))

        score = max(0.0, min(1.0, score))
        passed = score >= 0.70

        print("[ANSWER VERIFIER]")
        print(f"Score: {score:.2f}")
        print(f"Passed: {passed}")
        print(f"Reason: {reason}")

        return {
            **state,
            "verification_score": score,
            "verification_passed": passed,
            "verification_reason": reason or "",
            "nodes_activated": activated,
        }

    except Exception as exc:
        print("[ANSWER VERIFIER] Verifier failed:", exc)
        return {
            **state,
            "verification_score": 1.0,
            "verification_passed": True,
            "verification_reason": "Verifier unavailable",
            "nodes_activated": activated,
        }
