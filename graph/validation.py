"""
ValidationNode - filters retrieved context before synthesis.

The node keeps the existing state shape intact by rewriting the relevant
context list in-place. For numeric/stat queries it extracts structured rows
from retrieved table chunks and keeps only rows that satisfy the user's
condition, so the LLM never sees contradictory rows such as economy >= 7
for an "economy below 7" query.
"""

import re
from typing import Optional

from langchain_core.documents import Document

from graph.state import IPLAgentState
from rag.ingest import PLAYER_NAMES


CONTEXT_KEYS = [
    "batting_context",
    "bowling_context",
    "h2h_context",
    "venue_context",
    "form_context",
    "retrieved_chunks",
]

QUERY_TYPE_TO_KEYS = {
    "batting": ["batting_context"],
    "bowling": ["bowling_context"],
    "h2h": ["h2h_context"],
    "venue": ["venue_context"],
    "form": ["form_context"],
    "records": ["retrieved_chunks"],
    "team": ["retrieved_chunks"],
    "season": ["retrieved_chunks"],
    "prediction": ["h2h_context", "venue_context", "form_context", "batting_context"],
    "dream11": ["form_context", "batting_context"],
}

METRIC_PATTERNS = [
    (r"economy(?:\s+rate)?|econ", "economy", {"bowling"}),
    (r"wickets?|wkts", "wickets", {"bowling"}),
    (r"bowling\s+average|bowling\s+avg", "average", {"bowling"}),
    (r"strike\s+rate|sr", "strike_rate", {"batting", "bowling"}),
    (r"runs?|run\s+tally", "runs", {"batting"}),
    (r"batting\s+average|batting\s+avg|average|avg", "average", {"batting"}),
    (r"centur(?:y|ies)|100s", "centuries", {"batting"}),
    (r"fift(?:y|ies)|50s", "fifties", {"batting"}),
    (r"matches?|mat", "matches", {"batting", "bowling"}),
]

OPERATOR_PATTERN = (
    r"below|under|less\s+than|lower\s+than|<|"
    r"above|over|more\s+than|greater\s+than|>|"
    r"at\s+least|minimum|not\s+less\s+than|>=|"
    r"at\s+most|maximum|not\s+more\s+than|<=|"
    r"equal\s+to|equals|=|exactly"
)

OPERATOR_MAP = {
    "below": "<",
    "under": "<",
    "less than": "<",
    "lower than": "<",
    "<": "<",
    "above": ">",
    "over": ">",
    "more than": ">",
    "greater than": ">",
    ">": ">",
    "at least": ">=",
    "minimum": ">=",
    "not less than": ">=",
    ">=": ">=",
    "at most": "<=",
    "maximum": "<=",
    "not more than": "<=",
    "<=": "<=",
    "equal to": "==",
    "equals": "==",
    "=": "==",
    "exactly": "==",
}


# Known conflict pairs from Section 11 of the dataset.
KNOWN_CONFLICTS = {
    "virat kohli": {"runs": ["7263", "7084"]},
    "yuzvendra chahal": {"wickets": ["205", "187"]},
    "mi vs csk": {"matches": ["35", "33"]},
    "ms dhoni": {"matches": ["250", "240"]},
    "highest team score": {"score": ["287/3", "263/5"]},
    "best bowling figures": {"figures": ["6/12", "6/14"]},
}


def _normalise_operator(value: str) -> str:
    return OPERATOR_MAP[re.sub(r"\s+", " ", value.lower()).strip()]


def _extract_numeric_constraints(query: str) -> list[dict]:
    lowered = query.lower()
    constraints = []

    for metric_regex, field, sections in METRIC_PATTERNS:
        metric_then_value = re.compile(
            rf"(?:{metric_regex})(?:\s+\w+){{0,4}}\s+({OPERATOR_PATTERN})\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        value_then_metric = re.compile(
            rf"({OPERATOR_PATTERN})\s*(\d+(?:\.\d+)?)(?:\s+\w+){{0,3}}\s+(?:{metric_regex})",
            re.IGNORECASE,
        )

        for match in metric_then_value.finditer(lowered):
            constraints.append({
                "field": field,
                "operator": _normalise_operator(match.group(1)),
                "value": float(match.group(2)),
                "sections": sections,
            })

        for match in value_then_metric.finditer(lowered):
            constraints.append({
                "field": field,
                "operator": _normalise_operator(match.group(1)),
                "value": float(match.group(2)),
                "sections": sections,
            })

    unique = []
    seen = set()
    for constraint in constraints:
        key = (constraint["field"], constraint["operator"], constraint["value"], tuple(sorted(constraint["sections"])))
        if key not in seen:
            seen.add(key)
            unique.append(constraint)
    return unique


def _to_number(value: str) -> Optional[float]:
    cleaned = value.replace("*", "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return float(cleaned)
    return None


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_int(value: str) -> Optional[int]:
    number = _to_number(value)
    if number is None:
        return None
    return int(number)


def _parse_float(value: str) -> Optional[float]:
    return _to_number(value)


def _find_player_rows(lines: list[str]):
    player_lookup = {player.lower(): player for player in PLAYER_NAMES}
    for idx, line in enumerate(lines):
        player = player_lookup.get(line.lower())
        if player:
            yield idx, player


def _extract_bowling_rows(doc: Document) -> list[dict]:
    lines = _nonempty_lines(doc.page_content)
    rows = []

    for idx, player in _find_player_rows(lines):
        values = lines[idx + 1: idx + 9]
        if len(values) < 8:
            continue

        matches = _parse_int(values[1])
        wickets = _parse_int(values[2])
        average = _parse_float(values[3])
        economy = _parse_float(values[4])
        strike_rate = _parse_float(values[5])

        if None in [matches, wickets, average, economy, strike_rate]:
            continue

        rows.append({
            "section": "bowling",
            "player": player,
            "team": values[0],
            "matches": matches,
            "wickets": wickets,
            "average": average,
            "economy": economy,
            "strike_rate": strike_rate,
            "best": values[6],
            "type": values[7],
            "source": doc.metadata.get("source", "IPL Dataset"),
            "page": doc.metadata.get("page", ""),
        })

    return rows


def _extract_batting_rows(doc: Document) -> list[dict]:
    lines = _nonempty_lines(doc.page_content)
    rows = []

    for idx, player in _find_player_rows(lines):
        values = lines[idx + 1: idx + 10]
        if len(values) < 9:
            continue

        matches = _parse_int(values[1])
        runs = _parse_int(values[2])
        average = _parse_float(values[3])
        strike_rate = _parse_float(values[4])
        centuries = _parse_int(values[5])
        fifties = _parse_int(values[6])

        if None in [matches, runs, average, strike_rate, centuries, fifties]:
            continue

        rows.append({
            "section": "batting",
            "player": player,
            "team": values[0],
            "matches": matches,
            "runs": runs,
            "average": average,
            "strike_rate": strike_rate,
            "centuries": centuries,
            "fifties": fifties,
            "highest_score": values[7],
            "role": values[8],
            "source": doc.metadata.get("source", "IPL Dataset"),
            "page": doc.metadata.get("page", ""),
        })

    return rows


def _extract_rows(docs: list[Document], section: str) -> list[dict]:
    extractors = {
        "batting": _extract_batting_rows,
        "bowling": _extract_bowling_rows,
    }
    extractor = extractors.get(section)
    if not extractor:
        return []

    rows = []
    seen = set()
    for doc in docs:
        for row in extractor(doc):
            key = (row["section"], row["player"], row["team"])
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def _satisfies(value: float, operator: str, target: float) -> bool:
    if operator == "<":
        return value < target
    if operator == "<=":
        return value <= target
    if operator == ">":
        return value > target
    if operator == ">=":
        return value >= target
    if operator == "==":
        return value == target
    return False


def _row_satisfies(row: dict, constraints: list[dict]) -> bool:
    for constraint in constraints:
        if row["section"] not in constraint["sections"]:
            continue
        field = constraint["field"]
        if field not in row:
            return False
        if not _satisfies(float(row[field]), constraint["operator"], constraint["value"]):
            return False
    return True


def _constraints_for_section(constraints: list[dict], section: str) -> list[dict]:
    return [constraint for constraint in constraints if section in constraint["sections"]]


def _format_constraint(constraint: dict) -> str:
    return f"{constraint['field']} {constraint['operator']} {constraint['value']:g}"


def _format_rows(section: str, rows: list[dict]) -> str:
    if section == "bowling":
        lines = ["Validated bowling rows satisfying the query:"]
        lines.append("Player | Team | Mat | Wkts | Avg | Econ | SR | Best | Type")
        for row in rows:
            lines.append(
                f"{row['player']} | {row['team']} | {row['matches']} | {row['wickets']} | "
                f"{row['average']:g} | {row['economy']:g} | {row['strike_rate']:g} | "
                f"{row['best']} | {row['type']}"
            )
        return "\n".join(lines)

    if section == "batting":
        lines = ["Validated batting rows satisfying the query:"]
        lines.append("Player | Team | Mat | Runs | Avg | SR | 100s | 50s | HS | Role")
        for row in rows:
            lines.append(
                f"{row['player']} | {row['team']} | {row['matches']} | {row['runs']} | "
                f"{row['average']:g} | {row['strike_rate']:g} | {row['centuries']} | "
                f"{row['fifties']} | {row['highest_score']} | {row['role']}"
            )
        return "\n".join(lines)

    return ""


def _check_conflicts(docs: list[Document]) -> list[str]:
    conflicts_found = []

    for entity, fields in KNOWN_CONFLICTS.items():
        entity_docs = [
            doc for doc in docs
            if entity in doc.page_content.lower()
        ]
        if len(entity_docs) < 2:
            continue

        for field, known_values in fields.items():
            docs_with_value = {value: [] for value in known_values}
            for doc in entity_docs:
                for value in known_values:
                    if value in doc.page_content:
                        source = doc.metadata.get("source", "unknown source")
                        docs_with_value[value].append(source)

            populated = {value: sources for value, sources in docs_with_value.items() if sources}
            if len(populated) > 1:
                conflicts_found.append(
                    f"Conflict for {entity} ({field}): "
                    + " vs ".join(
                        f"{value} from {', '.join(sorted(set(sources)))}"
                        for value, sources in populated.items()
                    )
                )

    return conflicts_found


def _all_docs(state: IPLAgentState) -> list[Document]:
    docs = []
    for key in CONTEXT_KEYS:
        docs.extend(state.get(key, []))
    return docs


def _filter_numeric_context(state: IPLAgentState, constraints: list[dict]) -> tuple[dict, list[str]]:
    updates = {}
    notes = []

    query_type = state.get("query_type", "")
    target_keys = QUERY_TYPE_TO_KEYS.get(query_type, CONTEXT_KEYS)

    for section in ["bowling", "batting"]:
        section_constraints = _constraints_for_section(constraints, section)
        if not section_constraints:
            continue

        keys = [
            key for key in target_keys
            if key in {"bowling_context", "batting_context"}
            and key.startswith(section)
        ]
        if not keys:
            keys = [f"{section}_context"]

        docs = []
        for key in keys:
            docs.extend(state.get(key, []))

        rows = _extract_rows(docs, section)
        if not rows:
            continue

        filtered_rows = [row for row in rows if _row_satisfies(row, section_constraints)]
        constraint_text = ", ".join(_format_constraint(c) for c in section_constraints)

        if filtered_rows:
            content = _format_rows(section, filtered_rows)
            updates[f"{section}_context"] = [
                Document(
                    page_content=content,
                    metadata={
                        "source": "Validated IPL Dataset",
                        "section": section,
                        "validation": constraint_text,
                    },
                )
            ]
            notes.append(
                f"Filtered {section} context to {len(filtered_rows)} row(s) matching {constraint_text}."
            )
        else:
            updates[f"{section}_context"] = []
            notes.append(
                f"No {section} rows satisfy {constraint_text}; answer must be Information not available in dataset."
            )

    return updates, notes


def validation_node(state: IPLAgentState) -> IPLAgentState:
    """
    Validates retrieved context before final answer generation.
    """
    activated = state.get("nodes_activated", [])
    activated.append("ValidationNode")

    docs = _all_docs(state)
    notes = []
    updates = {}

    constraints = _extract_numeric_constraints(state.get("user_query", ""))
    if constraints:
        numeric_updates, numeric_notes = _filter_numeric_context(state, constraints)
        updates.update(numeric_updates)
        notes.extend(numeric_notes)

    docs_after_numeric = []
    temp_state = {**state, **updates}
    docs_after_numeric.extend(_all_docs(temp_state))

    conflicts = _check_conflicts(docs_after_numeric or docs)
    if conflicts:
        notes.append(
            "Contradictory retrieved values found. Do not choose a single value unless the context identifies it as validated: "
            + " | ".join(conflicts)
        )

    existing_notes = state.get("synthesised_context", "").strip()
    validation_notes = "\n".join(note for note in [existing_notes, *notes] if note)

    return {
        **state,
        **updates,
        "synthesised_context": validation_notes,
        "nodes_activated": activated,
    }
