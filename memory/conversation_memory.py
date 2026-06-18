from copy import deepcopy

from graph.state import IPLAgentState
from rag.retriever import detect_entities


MAX_TURNS = 5


class ConversationMemory:
    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        self.history = []
        self.entities = {
            "teams": [],
            "players": [],
            "venues": [],
            "matches": [],
        }

    def snapshot(self) -> tuple[list[dict], dict]:
        return deepcopy(self.history[-self.max_turns:]), deepcopy(self.entities)

    def load(self, history: list[dict], entities: dict):
        if history:
            self.history = deepcopy(history[-self.max_turns:])
        if not entities and history:
            entities = _entities_from_history(history)
        if entities:
            self.entities = _merge_entities(self.entities, entities)

    def remember(self, user_query: str, assistant_answer: str):
        self.history.append({
            "user": user_query,
            "assistant": assistant_answer,
        })
        self.history = self.history[-self.max_turns:]
        self.entities = _merge_entities(self.entities, _extract_conversation_entities(user_query))
        self.entities = _merge_entities(self.entities, _extract_answer_entities(assistant_answer))


_MEMORY = ConversationMemory()


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _merge_entities(base: dict, updates: dict) -> dict:
    merged = {
        "teams": _dedupe((base or {}).get("teams", []) + (updates or {}).get("teams", [])),
        "players": _dedupe((base or {}).get("players", []) + (updates or {}).get("players", [])),
        "venues": _dedupe((base or {}).get("venues", []) + (updates or {}).get("venues", [])),
        "matches": _dedupe((base or {}).get("matches", []) + (updates or {}).get("matches", [])),
    }
    return merged


def _extract_conversation_entities(text: str) -> dict:
    detected = detect_entities(text or "")
    teams = detected.get("teams", [])
    matches = []
    if len(teams) >= 2:
        matches.append(f"{teams[-2]} vs {teams[-1]}")
    return {
        "teams": teams,
        "players": detected.get("players", []),
        "venues": detected.get("venues", []),
        "matches": matches,
    }


def _extract_answer_entities(text: str) -> dict:
    if not text:
        return {"teams": [], "players": [], "venues": [], "matches": []}

    compact = " ".join(str(text).split())
    if "[Chunk" in compact or len(compact.split()) > 100:
        return {"teams": [], "players": [], "venues": [], "matches": []}

    if "Sources:" in compact:
        compact = compact.split("Sources:", 1)[0]

    return _extract_conversation_entities(compact)


def _entities_from_history(history: list[dict]) -> dict:
    entities = {"teams": [], "players": [], "venues": [], "matches": []}
    for item in history[-MAX_TURNS:]:
        user_text = item.get("user") or item.get("query") or ""
        assistant_text = item.get("assistant") or item.get("answer") or ""
        entities = _merge_entities(entities, _extract_conversation_entities(user_text))
        entities = _merge_entities(entities, _extract_answer_entities(assistant_text))
    return entities


def _latest(values: list[str]) -> str:
    return values[-1] if values else ""


def _resolve_pronouns(query: str, entities: dict) -> str:
    lowered = f" {query.lower()} "
    resolved = query

    latest_player = _latest(entities.get("players", []))
    latest_team = _latest(entities.get("teams", []))
    latest_match = _latest(entities.get("matches", []))

    if latest_player and any(token in lowered for token in [" his ", " him ", " he "]):
        resolved = f"{resolved} {latest_player}"

    if latest_team and any(token in lowered for token in [" their ", " they ", " them "]):
        resolved = f"{resolved} {latest_team}"

    if latest_match and any(token in lowered for token in [" this match ", " the match ", " that match "]):
        resolved = f"{resolved} {latest_match}"

    return resolved


def _summary(entities: dict) -> str:
    parts = []
    for key in ["teams", "players", "venues", "matches"]:
        values = entities.get(key, [])
        if values:
            parts.append(f"{key}: {', '.join(values[-3:])}")
    return " | ".join(parts)


def memory_node(state: IPLAgentState) -> IPLAgentState:
    history, stored_entities = _MEMORY.snapshot()
    provided_history = state.get("chat_history", []) or []
    if provided_history:
        state_history = provided_history[-MAX_TURNS:]
        state_entities = state.get("conversation_entities", {}) or _entities_from_history(state_history)
    else:
        state_history = history
        state_entities = state.get("conversation_entities", {}) or stored_entities
    _MEMORY.load(state_history, state_entities)

    query = state.get("user_query", "")
    merged_entities = _merge_entities(state_entities, _extract_conversation_entities(query))
    resolved_query = _resolve_pronouns(query, merged_entities)
    merged_entities = _merge_entities(merged_entities, _extract_conversation_entities(resolved_query))

    activated = state.get("nodes_activated", [])
    activated.append("Memory")

    return {
        **state,
        "user_query": resolved_query,
        "chat_history": state_history[-MAX_TURNS:],
        "conversation_entities": merged_entities,
        "conversation_summary": _summary(merged_entities),
        "nodes_activated": activated,
    }


def memory_update_node(state: IPLAgentState) -> IPLAgentState:
    _MEMORY.remember(state.get("user_query", ""), state.get("final_answer", ""))
    history, entities = _MEMORY.snapshot()

    activated = state.get("nodes_activated", [])
    activated.append("MemoryUpdate")

    return {
        **state,
        "chat_history": history,
        "conversation_entities": entities,
        "conversation_summary": _summary(entities),
        "nodes_activated": activated,
    }
