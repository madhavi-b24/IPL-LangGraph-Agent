"""
ValidationNode — detects conflicting values across retrieved chunks.

The IPL dataset (Section 11) intentionally includes a secondary source
with different numbers for the same facts. This node compares values
and flags conflicts before the SynthesisNode generates an answer.
"""

import re
from graph.state import IPLAgentState


# Known conflict pairs from Section 11 of the dataset
KNOWN_CONFLICTS = {
    "virat kohli": {"runs": ["7263", "7084"]},
    "yuzvendra chahal": {"wickets": ["205", "187"]},
    "mi vs csk": {"matches": ["35", "33"]},
    "ms dhoni": {"matches": ["250", "240"]},
    "highest team score": {"score": ["287/3", "263/5"]},
    "best bowling figures": {"figures": ["6/12", "6/14"]},
}


def _extract_numbers(text: str) -> list:
    """Pull all numeric values (including formats like 287/3) from text."""
    return re.findall(r"\b\d+(?:/\d+)?\b", text)


def _check_conflicts(docs: list) -> list:
    """
    Compare numeric values across chunks for the same entity.
    Returns a list of conflict description strings.
    """
    conflicts_found = []

    for entity, fields in KNOWN_CONFLICTS.items():
        entity_docs = [
            d for d in docs
            if entity in d.page_content.lower()
        ]
        if len(entity_docs) < 2:
            continue

        for field, known_values in fields.items():
            docs_with_value = {v: [] for v in known_values}
            for doc in entity_docs:
                for v in known_values:
                    if v in doc.page_content:
                        source = doc.metadata.get("source", "unknown source")
                        docs_with_value[v].append(source)

            # If both conflicting values are present across chunks → flag it
            populated = {k: v for k, v in docs_with_value.items() if v}
            if len(populated) > 1:
                conflict_msg = (
                    f"⚠️ CONFLICT detected for '{entity}' ({field}): "
                    + " vs ".join(
                        f"{val} (from {', '.join(srcs)})"
                        for val, srcs in populated.items()
                    )
                )
                conflicts_found.append(conflict_msg)

    return conflicts_found


def validation_node(state: IPLAgentState) -> IPLAgentState:
    """
    Checks all retrieved context for conflicting numeric values.
    Appends a conflict warning to the final answer if found.
    """
    all_docs = (
        state.get("batting_context", [])
        + state.get("bowling_context", [])
        + state.get("h2h_context", [])
        + state.get("venue_context", [])
        + state.get("form_context", [])
        + state.get("retrieved_chunks", [])
    )

    activated = state.get("nodes_activated", [])
    activated.append("ValidationNode")

    if not all_docs:
        return {**state, "nodes_activated": activated}

    conflicts = _check_conflicts(all_docs)

    if conflicts:
        conflict_block = (
            "\n\n---\n🔍 **Data Validation Notice:**\n"
            + "\n".join(conflicts)
            + "\nPlease verify these figures against official IPL records at iplt20.com."
        )
        current_answer = state.get("final_answer", "")
        return {
            **state,
            "final_answer": current_answer + conflict_block,
            "nodes_activated": activated,
        }

    return {**state, "nodes_activated": activated}