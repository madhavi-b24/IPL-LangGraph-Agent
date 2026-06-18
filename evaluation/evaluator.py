from graph.state import IPLAgentState


def _clamp_score(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(value)
    except Exception:
        return minimum
    if value != value:
        return minimum
    return max(min(value, maximum), minimum)


def _retrieval_score(state: IPLAgentState) -> float:
    docs = []
    for key in [
        "batting_context",
        "bowling_context",
        "h2h_context",
        "venue_context",
        "form_context",
        "retrieved_chunks",
    ]:
        docs.extend(state.get(key, []) or [])

    scores = []
    for doc in docs:
        metadata = getattr(doc, "metadata", None) or {}
        score = metadata.get("_cross_encoder_score", metadata.get("_retrieval_score"))
        if score is not None:
            scores.append(_clamp_score(score))

    if scores:
        return _clamp_score(sum(scores) / len(scores))
    return _clamp_score(min(len(docs), 6) / 6.0 if docs else 0.0)


def _quality_level(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    if score >= 0.70:
        return "Good"
    if score >= 0.50:
        return "Fair"
    return "Weak"


def evaluation_node(state: IPLAgentState) -> IPLAgentState:
    retrieval_score = _retrieval_score(state)
    confidence_score = _clamp_score(state.get("confidence_score", state.get("confidence", 0.0) or 0.0))
    verification_score = _clamp_score(state.get("verification_score", 1.0) or 1.0)
    hallucination_score = _clamp_score(state.get("hallucination_score", 0.0) or 0.0)

    overall_score = _clamp_score(
        0.30 * retrieval_score
        + 0.30 * confidence_score
        + 0.20 * verification_score
        + 0.20 * (1.0 - hallucination_score)
    )
    level = _quality_level(overall_score)

    print("[EVALUATION]")
    print(f"Retrieval: {retrieval_score:.2f}")
    print(f"Confidence: {confidence_score:.2f}")
    print(f"Verification: {verification_score:.2f}")
    print(f"Hallucination: {hallucination_score:.2f}")
    print(f"Overall: {overall_score:.2f}")
    print(f"Level: {level}")

    activated = state.get("nodes_activated", [])
    activated.append("Evaluation")

    evaluation = {
        "retrieval_score": retrieval_score,
        "confidence_score": confidence_score,
        "verification_score": verification_score,
        "hallucination_score": hallucination_score,
        "overall_score": overall_score,
        "level": level,
    }

    return {
        **state,
        "evaluation": evaluation,
        "overall_quality_score": overall_score,
        "quality_level": level,
        "nodes_activated": activated,
    }
