from typing import Optional
from graph.state import IPLAgentState


def _evidence_strength_label(count: int) -> str:
    if count >= 8:
        return "Strong"
    if 4 <= count <= 7:
        return "Medium"
    if 1 <= count < 4:
        return "Weak"
    return "Unknown"


def prediction_confidence_node(state: IPLAgentState) -> IPLAgentState:
    activated = state.get("nodes_activated", [])
    activated.append("PredictionConfidence")

    try:
        qtype = state.get("query_type", "")
        if qtype not in {"prediction", "dream11", "prediction_query", "dream11_query"}:
            # Only apply to prediction/dream11 queries
            return {
                **state,
                "prediction_confidence": None,
                "evidence_strength": "Unknown",
                "supporting_sources": 0,
                "nodes_activated": activated,
            }

        # Gather inputs with safe defaults
        retrieval = float(state.get("confidence_score", state.get("confidence", 0.0) or 0.0))
        verification = float(state.get("verification_score", 1.0) or 1.0)
        hallucination = float(state.get("hallucination_score", 0.0) or 0.0)
        sources = state.get("source_attribution") or state.get("supporting_sources") or []
        if isinstance(sources, int):
            source_count = int(sources)
        else:
            try:
                source_count = len(sources)
            except Exception:
                source_count = 0

        # Compute a combined confidence (0-1)
        # weights: retrieval 0.40, verification 0.35, sources 0.15, hallucination_penalty 0.10
        sources_score = min(source_count, 10) / 10.0
        hallucination_penalty = max(0.0, 1.0 - hallucination)  # higher is better

        combined = (
            0.40 * retrieval
            + 0.35 * verification
            + 0.15 * sources_score
            + 0.10 * hallucination_penalty
        )

        prediction_confidence = max(0.0, min(1.0, combined)) * 100.0
        evidence_strength = _evidence_strength_label(source_count)

        print("[PREDICTION CONFIDENCE]")
        print(f"Prediction Confidence: {prediction_confidence:.1f}%")
        print(f"Evidence Strength: {evidence_strength}")
        print(f"Supporting Sources: {source_count}")

        return {
            **state,
            "prediction_confidence": prediction_confidence,
            "evidence_strength": evidence_strength,
            "supporting_sources": source_count,
            "nodes_activated": activated,
        }

    except Exception as exc:
        print("[PREDICTION CONFIDENCE] Calibration failed:", exc)
        return {
            **state,
            "prediction_confidence": 0,
            "evidence_strength": "Unknown",
            "supporting_sources": 0,
            "nodes_activated": activated,
        }
