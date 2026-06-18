import re
from typing import Iterable

from graph.state import IPLAgentState
from rag.retriever import detect_entities, flatten_entities


def _clamp_score(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(value)
    except Exception:
        return minimum
    if value != value:  # NaN
        return minimum
    return max(min(value, maximum), minimum)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _normalize_entities(entities: Iterable[str]) -> set[str]:
    return {str(entity).strip().lower() for entity in entities if str(entity).strip()}


def _extract_entities_from_text(text: str) -> set[str]:
    if not text:
        return set()

    detected = detect_entities(text)
    return _normalize_entities(flatten_entities(detected))


def _collect_context_text(state: IPLAgentState) -> str:
    content_parts = []
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
                content_parts.append(str(doc.page_content or ""))
                metadata = getattr(doc, "metadata", None) or {}
                for field in ["section", "source", "teams", "players", "venues", "tags"]:
                    value = metadata.get(field)
                    if value:
                        content_parts.append(str(value))
            except Exception:
                continue

    return "\n".join(content_parts)


def _entity_support_score(answer_entities: set[str], context_entities: set[str]) -> tuple[float, str]:
    if not answer_entities:
        return 0.0, ""

    unsupported = answer_entities.difference(context_entities)
    if not unsupported:
        return 0.0, ""

    unsupported_ratio = len(unsupported) / max(len(answer_entities), 1)
    score = _clamp_score(unsupported_ratio * 0.75)
    reason = f"Unsupported entities: {', '.join(sorted(unsupported))}"
    return score, reason


def _coverage_score(answer_text: str, context_text: str) -> tuple[float, str]:
    answer_words = _word_count(answer_text)
    context_words = max(_word_count(context_text), 1)
    ratio = answer_words / context_words

    if ratio <= 1.5:
        return 0.0, ""

    score = _clamp_score((ratio - 1.5) * 0.15)
    if score <= 0.0:
        return 0.0, ""

    reason = (
        "Answer length exceeds retrieved evidence by a large margin"
        if ratio >= 3.0
        else "Answer is longer than retrieved evidence"
    )
    return score, reason


def _confidence_penalty(confidence_score: float) -> tuple[float, str]:
    if confidence_score >= 0.45:
        return 0.0, ""

    score = _clamp_score((0.45 - confidence_score) * 0.55)
    reason = "Low confidence score"
    return score, reason


def _source_score(answer_text: str, source_attribution: list[dict]) -> tuple[float, str]:
    if not answer_text or source_attribution:
        return 0.0, ""

    return 0.30, "Missing source attribution"


def hallucination_guard_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("HallucinationGuard")

    try:
        final_answer = str(state.get("final_answer", "")).strip()
        confidence_score = float(state.get("confidence_score", 0.0))
        source_attribution = state.get("source_attribution", []) or []

        context_text = _collect_context_text(state)
        answer_entities = _extract_entities_from_text(final_answer)
        context_entities = _extract_entities_from_text(context_text)

        entity_score, entity_reason = _entity_support_score(answer_entities, context_entities)
        coverage_score, coverage_reason = _coverage_score(final_answer, context_text)
        confidence_score_adj, confidence_reason = _confidence_penalty(confidence_score)
        source_score, source_reason = _source_score(final_answer, source_attribution)

        score = _clamp_score(entity_score + coverage_score + confidence_score_adj + source_score)
        reasons = [r for r in [entity_reason, coverage_reason, confidence_reason, source_reason] if r]
        reason_text = "; ".join(reasons) if reasons else "No hallucination signals detected."
        flag = score >= 0.7

        print("[HALLUCINATION GUARD]")
        print(f"Score: {score:.2f}")
        print(f"Flag: {flag}")
        print(f"Reason: {reason_text}")

        return {
            **state,
            "hallucination_score": score,
            "hallucination_flag": flag,
            "hallucination_reason": reason_text,
            "nodes_activated": activated,
        }
    except Exception as exc:
        print("[HALLUCINATION GUARD] Guard failed:", exc)
        return {
            **state,
            "hallucination_score": 0.0,
            "hallucination_flag": False,
            "hallucination_reason": "Guard failed; defaults applied.",
            "nodes_activated": activated,
        }
