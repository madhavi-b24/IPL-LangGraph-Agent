import re
from typing import Optional
from rag.ingest import TEAM_ALIASES, PLAYER_NAMES, VENUE_NAMES, load_vectorstore

# Load once at module level (shared across all nodes)
_vectorstore = None

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "how", "in", "is", "it", "me", "of", "on", "or",
    "the", "to", "what", "which", "who", "will", "with",
}

SECTION_TEXT_MARKERS = {
    "batting": ["player batting statistics", "battingstatsnode", " runs ", " strike rate ", " sr "],
    "bowling": ["player bowling statistics", "bowlingstatsnode", " wickets ", " wkts ", " economy ", " econ "],
    "venue": ["venue & pitch", "venuenode", "stadium", "pitch", "ground"],
    "h2h": ["head-to-head", "h2hnode", "matchup", " vs "],
    "form": ["recent form", "formnode", "last 5", "form trend"],
    "records": ["records & milestones", "recordsnode", "record", "milestone"],
    "team": ["team profiles", "teamprofilenode", "captain", "coach", "home venue"],
    "season": ["season-wise", "trendnode", "2019", "2024"],
}

BOWLING_METRIC_TERMS = {"economy", "econ", "wicket", "wickets", "wkts", "average", "avg", "best", "figures", "sr"}
BATTING_METRIC_TERMS = {"run", "runs", "average", "avg", "strike", "rate", "sr", "century", "centuries", "fifty", "fifties", "50s", "100s"}


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.lower()
    if len(phrase) <= 4 and phrase.isalnum():
        return re.search(rf"\b{re.escape(phrase)}\b", text) is not None
    return phrase in text


def expand_query(query: str) -> str:
    """Expand IPL abbreviations so embeddings see canonical team names."""
    expanded = query
    for short_name, full_name in TEAM_ALIASES.items():
        pattern = rf"\b{re.escape(short_name)}\b"
        expanded = re.sub(
            pattern,
            f"{short_name} {full_name}",
            expanded,
            flags=re.IGNORECASE,
        )
    return expanded


def detect_entities(query: str) -> dict:
    """Detect team, player, and venue entities from the user query."""
    expanded = expand_query(query)
    lowered = expanded.lower()

    teams = []
    for short_name, full_name in TEAM_ALIASES.items():
        if _contains_phrase(lowered, short_name) or _contains_phrase(lowered, full_name):
            teams.append(full_name)

    players = []
    for name in PLAYER_NAMES:
        last_name = name.split()[-1]
        if _contains_phrase(lowered, name) or _contains_phrase(lowered, last_name):
            players.append(name)

    venues = []
    for name in VENUE_NAMES:
        if _contains_phrase(lowered, name):
            venues.append(name)

    return {
        "teams": sorted(set(teams)),
        "players": sorted(set(players)),
        "venues": sorted(set(venues)),
    }


def flatten_entities(entities: dict) -> list:
    return entities.get("players", []) + entities.get("teams", []) + entities.get("venues", [])


def _query_terms(query: str) -> set:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 2 and term not in STOPWORDS
    }


def _doc_key(doc) -> tuple:
    metadata = doc.metadata or {}
    return (
        metadata.get("source", ""),
        metadata.get("page", ""),
        re.sub(r"\s+", " ", doc.page_content).strip()[:220],
    )


def _metadata_sections(metadata: dict) -> set:
    sections = set()
    section = str(metadata.get("section", "")).strip().lower()
    if section:
        sections.add(section)
    tags = str(metadata.get("tags", "")).lower()
    sections.update(tag.strip() for tag in tags.split(",") if tag.strip())
    return sections


def _section_matches(doc, section: str) -> bool:
    metadata = doc.metadata or {}
    primary_section = str(metadata.get("section", "")).strip().lower()
    if primary_section:
        return primary_section == section

    if section in _metadata_sections(metadata):
        return True

    lowered = f" {doc.page_content.lower()} "
    return any(marker in lowered for marker in SECTION_TEXT_MARKERS.get(section, []))


def _has_stat_row_shape(doc, section: str) -> bool:
    text = " ".join(doc.page_content.lower().split())
    player_hits = sum(1 for player in PLAYER_NAMES if player.lower() in text)
    decimal_hits = len(re.findall(r"\b\d+\.\d+\b", text))

    if section == "bowling":
        return (
            "player bowling statistics" in text
            or " econ " in f" {text} "
            or (player_hits >= 2 and decimal_hits >= 3 and any(term in text for term in ["wicket", "wkts", "bowling"]))
        )

    if section == "batting":
        return (
            "player batting statistics" in text
            or " strike rate " in text
            or " sr " in f" {text} "
            or (player_hits >= 2 and decimal_hits >= 3 and any(term in text for term in ["runs", "opener", "bat"]))
        )

    return True


def _is_query_compatible(doc, section: str, query_terms: set) -> bool:
    if not _section_matches(doc, section):
        return False

    if section == "bowling" and query_terms.intersection(BOWLING_METRIC_TERMS):
        return _has_stat_row_shape(doc, section)

    if section == "batting" and query_terms.intersection(BATTING_METRIC_TERMS):
        return _has_stat_row_shape(doc, section)

    return True


def _score_doc(doc, distance: float, section: str, entities: dict, query_terms: set) -> float:
    text = doc.page_content.lower()
    metadata = doc.metadata or {}
    metadata_text = " ".join(
        str(metadata.get(field, "")).lower()
        for field in ["section", "tags", "teams", "players", "venues"]
    )
    searchable_text = f"{text} {metadata_text}"

    vector_score = 1 / (1 + distance) if isinstance(distance, (int, float)) else 0.0
    score = vector_score

    if metadata.get("section") == section:
        score += 0.85
    elif section in _metadata_sections(metadata):
        score += 0.45

    tags = metadata.get("tags", "")
    if section in tags:
        score += 0.20

    if not _is_query_compatible(doc, section, query_terms):
        score -= 1.25

    for entity in flatten_entities(entities):
        if entity.lower() in searchable_text:
            score += 0.25

    if section == "h2h" and len(entities.get("teams", [])) >= 2:
        if " vs " in text or "head-to-head" in text or "matchup" in text:
            score += 0.35

    if section == "batting" and {"opener", "strike", "rate"}.issubset(query_terms):
        if "opener" in text and (" sr " in f" {text} " or "strike rate" in text):
            score += 0.30

    if section == "team" and ("captain" in query_terms or "captains" in query_terms):
        if "captain" in text:
            score += 0.25

    matched_terms = sum(1 for term in query_terms if term in searchable_text)
    score += min(matched_terms * 0.04, 0.40)

    return score


def _search_with_scores(vs, query: str, k: int, metadata_filter: Optional[dict] = None):
    try:
        return vs.similarity_search_with_score(
            query,
            k=k,
            filter=metadata_filter,
        )
    except Exception as exc:
        print("Similarity score search error:", exc)
        try:
            docs = vs.similarity_search(query, k=k, filter=metadata_filter)
        except TypeError:
            docs = vs.similarity_search(query, k=k)
        return [(doc, None) for doc in docs]


def _rank_candidates(candidates, section: str, entities: dict, query: str):
    query_terms = _query_terms(query)
    ranked = []
    seen = set()

    for doc, distance, source in candidates:
        if not _is_query_compatible(doc, section, query_terms):
            continue

        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)

        final_score = _score_doc(doc, distance, section, entities, query_terms)
        ranked.append((doc, distance, source, final_score))

    ranked.sort(key=lambda item: item[3], reverse=True)
    return ranked


def retrieve(query: str, section: str, k: int = 6, entities: Optional[dict] = None):
    expanded_query = expand_query(query)
    detected_entities = entities or detect_entities(query)
    metadata_filter = {"section": section}
    fetch_k = max(k * 3, 12)

    print("\n========== RETRIEVAL ==========")
    print("Query:", query)
    print("Expanded Query:", expanded_query)
    print("Detected Entity:", detected_entities)
    print("Metadata Filter:", metadata_filter)

    vs = get_vectorstore()
    candidates = []

    filtered_results = _search_with_scores(
        vs,
        expanded_query,
        k=fetch_k,
        metadata_filter=metadata_filter,
    )
    print("Filtered candidates:", len(filtered_results))
    candidates.extend((doc, distance, "metadata") for doc, distance in filtered_results)

    if len(filtered_results) < k:
        print("Metadata retrieval returned too few chunks; running section-compatible semantic fallback.")
        broad_results = _search_with_scores(vs, expanded_query, k=fetch_k * 2)
        compatible_fallback = [
            (doc, distance)
            for doc, distance in broad_results
            if _section_matches(doc, section)
        ]
        print("Section-compatible fallback candidates:", len(compatible_fallback))
        candidates.extend((doc, distance, "fallback") for doc, distance in compatible_fallback)

    ranked = _rank_candidates(candidates, section, detected_entities, expanded_query)
    docs = [doc for doc, _, _, _ in ranked[:k]]

    print("Retrieved chunks:", len(docs))
    for idx, (doc, distance, source, score) in enumerate(ranked[:k], start=1):
        metadata = doc.metadata or {}
        distance_label = f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"
        similarity = 1 / (1 + distance) if isinstance(distance, (int, float)) else None
        similarity_label = f"{similarity:.4f}" if similarity is not None else "n/a"
        preview = " ".join(doc.page_content.split())[:220]
        print(
            f"{idx}. source={source} section={metadata.get('section')} "
            f"distance={distance_label} similarity={similarity_label} rank_score={score:.4f} "
            f"page={metadata.get('page')} preview={preview}"
        )

    print("==============================\n")
    return docs
