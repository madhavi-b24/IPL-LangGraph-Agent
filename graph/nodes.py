import os
import re
from langchain_groq import ChatGroq
from graph.state import IPLAgentState
from rag.retriever import detect_entities, expand_query, flatten_entities, retrieve
from tools.tool_router import ToolRouter

_llm = None
_rewriter_llm = None
_tool_router = None


def _get_llm():
    """Lazily instantiate the LLM client to avoid requiring API keys at import time."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            _llm = None
        else:
            _llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.1)
    return _llm


def _get_rewriter_llm():
    global _rewriter_llm
    if _rewriter_llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            _rewriter_llm = None
        else:
            _rewriter_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return _rewriter_llm

def rewrite_query_node(state: IPLAgentState) -> IPLAgentState:
    """Rewrites user query for improved semantic retrieval.
    
    Expands abbreviations, disambiguates intent, normalizes language.
    Stores rewritten query in state["rewritten_query"].
    Falls back to original query if rewriting fails.
    """
    original_query = state["user_query"]
    
    try:
        rewriter = _get_rewriter_llm()
        if rewriter is None:
            # No API key or LLM unavailable — skip rewriting
            rewritten = original_query
        else:
            prompt = f"""You are an IPL query rewriting assistant.

Rewrite the query for semantic retrieval. Preserve the original meaning.

Rules:
- Expand cricket abbreviations (e.g., SR → strike rate, econ → economy rate).
- Expand team abbreviations to full names (e.g., MI → Mumbai Indians).
- Add context keywords relevant to IPL.
- Rephrase unclear or vague phrasing.
- Keep the core intent unchanged.
- Return ONLY the rewritten query, no explanation.

Original Query: {original_query}

Rewritten Query:"""

            response = rewriter.invoke(prompt)
            rewritten = getattr(response, 'content', '') or str(response)
            rewritten = rewritten.strip()

            # Fallback if response is empty
            if not rewritten or len(rewritten) < 3:
                rewritten = original_query
    except Exception as e:
        print(f"Query rewriting failed: {e}. Using original query.")
        rewritten = original_query
    
    print(f"[RewriteQueryNode] original_query={original_query!r} rewritten_query={rewritten!r}")
    activated = state.get("nodes_activated", [])
    activated.append("RewriteNode")
    
    return {
        **state,
        "rewritten_query": rewritten,
        "nodes_activated": activated,
    }


def _get_tool_router() -> ToolRouter:
    global _tool_router
    if _tool_router is None:
        _tool_router = ToolRouter()
    return _tool_router


def _resolve_retrieval_query(state: IPLAgentState) -> str:
    original_query = state["user_query"]
    rewritten_query = state.get("rewritten_query") or original_query
    final_query = rewritten_query
    print(f"[RetrievalQuery] original_query={original_query!r} rewritten_query={rewritten_query!r} final_query={final_query!r}")
    return final_query


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _clamp_score(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(value)
    except Exception:
        return minimum
    if value != value:  # NaN
        return minimum
    return max(min(value, maximum), minimum)


def _confidence_level(score: float) -> str:
    if score < 0.50:
        return "LOW"
    if score < 0.80:
        return "MEDIUM"
    return "HIGH"


def _collect_source_attribution(docs: list) -> list[dict]:
    try:
        seen = set()
        attribution = []
        for doc in docs:
            metadata = doc.metadata or {}
            section = str(metadata.get("section", "unknown"))
            page = str(metadata.get("page", "n/a"))
            source = str(metadata.get("source", "IPL Dataset"))
            key = (section, page, source)
            if key in seen:
                continue
            seen.add(key)
            attribution.append({
                "section": section,
                "page": page,
                "source": source,
                "metadata": {
                    k: v for k, v in metadata.items()
                    if k not in {"_retrieval_score", "_cross_encoder_score"}
                },
            })
        return attribution
    except Exception:
        return []


def _compute_confidence(state: IPLAgentState, docs: list) -> tuple[float, str]:
    try:
        tool_confidence = _clamp_score(state.get("tool_confidence", 0.0))
        retrieval_scores = [
            _clamp_score(doc.metadata.get("_retrieval_score", 0.0))
            for doc in docs
            if getattr(doc, "metadata", None) is not None
        ]
        rerank_scores = [
            _clamp_score(doc.metadata.get("_cross_encoder_score", doc.metadata.get("_retrieval_score", 0.0)))
            for doc in docs
            if getattr(doc, "metadata", None) is not None
        ]

        retrieval_score = (
            sum(retrieval_scores) / len(retrieval_scores)
            if retrieval_scores
            else 0.5
        )
        cross_encoder_score = (
            sum(rerank_scores) / len(rerank_scores)
            if rerank_scores
            else retrieval_score
        )

        support_ratio = min(len(docs), 6) / 6.0 if docs else 0.0

        confidence_score = (
            0.40 * retrieval_score
            + 0.25 * cross_encoder_score
            + 0.20 * tool_confidence
            + 0.15 * support_ratio
        )
        confidence_score = _clamp_score(confidence_score)
        confidence_level = _confidence_level(confidence_score)
        return confidence_score, confidence_level
    except Exception:
        return 0.5, "MEDIUM"


# ── NODE 1: RouterNode ──────────────────────────────────────────────────────
def router_node(state: IPLAgentState) -> IPLAgentState:
    """Classifies query type and extracts team/player entity names."""
    # Prefer rewritten query for routing when available
    raw_query = state.get("rewritten_query") or state["user_query"]
    query = expand_query(raw_query).lower()
    detected_entities = detect_entities(raw_query)
    entities = flatten_entities(detected_entities)

    team_count = len(detected_entities.get("teams", []))
    has_matchup = bool(
        team_count >= 2
        or _has_any(query, [" vs ", " versus ", " v "])
        or re.search(r"\b[a-z]{2,4}\s+vs\s+[a-z]{2,4}\b", query)
    )

    dream11_terms = ["dream11", "fantasy", "captain pick", "vice captain", "xi", "team pick"]
    prediction_terms = ["predict", "who will win", "likely to win", "winner", "win probability", "favoured", "favored"]
    h2h_terms = ["head-to-head", "h2h", "played each other", "matchup", "against each other"]
    venue_terms = ["venue", "pitch", "stadium", "ground", "dew", "chinnaswamy", "wankhede", "chepauk", "eden gardens"]
    form_terms = ["form", "last 5", "recent", "this season", "current form", "trend"]
    bowling_terms = ["bowl", "bowler", "bowlers", "bowling", "wicket", "wickets", "wkts", "economy", "econ", "spell", "figures"]
    batting_terms = ["bat", "batter", "batters", "batsman", "runs", "run tally", "average", "strike rate", "sr", "century", "fifty", "opener"]
    records_terms = ["record", "records", "highest", "most", "fastest", "best figures", "milestone", "highest team total", "highest chase"]
    team_terms = ["captain", "coach", "home venue", "squad", "title", "titles", "team profile", "tell me about"]
    season_terms = ["season-wise", "2019", "2020", "2021", "2022", "2023", "2024", "consistent"]

    if _has_any(query, dream11_terms):
        query_type = "dream11"
    elif has_matchup and _has_any(query, prediction_terms):
        query_type = "prediction"
    elif has_matchup and (_has_any(query, h2h_terms) or " vs " in query or " between " in query):
        query_type = "h2h"
    elif _has_any(query, bowling_terms):
        query_type = "bowling"
    elif _has_any(query, batting_terms):
        query_type = "batting"
    elif _has_any(query, venue_terms):
        query_type = "venue"
    elif _has_any(query, form_terms):
        query_type = "form"
    elif _has_any(query, records_terms):
        query_type = "records"
    elif _has_any(query, team_terms):
        query_type = "team"
    elif _has_any(query, season_terms):
        query_type = "season"
    elif "team" in query:
        query_type = "team"
    else:
        query_type = "records"  # default fallback

    activated = state.get("nodes_activated", [])
    activated.append("RouterNode")
    print(f"[RouterNode] routing query='{raw_query}' expanded='{query}' type_guess='{query_type}' entities={detected_entities}")

    return {
        **state,
        "query_type": query_type,
        "entities": entities,
        "nodes_activated": activated,
    }


def tool_router_node(state: IPLAgentState) -> IPLAgentState:
    raw_query = state.get("rewritten_query") or state["user_query"]
    router = _get_tool_router()
    selection = router.select_tool(raw_query)
    selected_tool = selection.get("selected_tool")
    confidence = selection.get("confidence", 0.0)

    activated = state.get("nodes_activated", [])
    activated.append("ToolRouterNode")
    print(f"[TOOL ROUTER] Selected Tool: {selected_tool!r}")
    print(f"[TOOL ROUTER] Confidence: {confidence:.4f}")
    print(f"[TOOL ROUTER] Tool Executed: {selected_tool!r}")

    return {
        **state,
        "selected_tool": selected_tool,
        "tool_confidence": confidence,
        "nodes_activated": activated,
    }


# ── NODE 2: BattingStatsNode ────────────────────────────────────────────────
def batting_stats_node(state: IPLAgentState) -> IPLAgentState:
    query = _resolve_retrieval_query(state)
    docs = retrieve(query, section="batting")
    activated = state.get("nodes_activated", [])
    activated.append("BattingStatsNode")
    return {**state, "batting_context": docs, "nodes_activated": activated}


# ── NODE 3: BowlingStatsNode ────────────────────────────────────────────────
def bowling_stats_node(state: IPLAgentState) -> IPLAgentState:
    query = _resolve_retrieval_query(state)
    docs = retrieve(query, section="bowling")
    activated = state.get("nodes_activated", [])
    activated.append("BowlingStatsNode")
    return {**state, "bowling_context": docs, "nodes_activated": activated}


# ── NODE 4: VenueNode ───────────────────────────────────────────────────────
def venue_node(state: IPLAgentState) -> IPLAgentState:
    query = _resolve_retrieval_query(state)
    docs = retrieve(query, section="venue")
    activated = state.get("nodes_activated", [])
    activated.append("VenueNode")
    return {**state, "venue_context": docs, "nodes_activated": activated}


# ── NODE 5: H2HNode ─────────────────────────────────────────────────────────
def h2h_node(state: IPLAgentState) -> IPLAgentState:
    query = _resolve_retrieval_query(state)
    docs = retrieve(query, section="h2h")
    activated = state.get("nodes_activated", [])
    activated.append("H2HNode")
    return {**state, "h2h_context": docs, "nodes_activated": activated}


# ── NODE 6: FormNode ────────────────────────────────────────────────────────
def form_node(state: IPLAgentState) -> IPLAgentState:
    query = _resolve_retrieval_query(state)
    docs = retrieve(query, section="form")
    activated = state.get("nodes_activated", [])
    activated.append("FormNode")
    return {**state, "form_context": docs, "nodes_activated": activated}


# ── NODE 7: RecordsNode ─────────────────────────────────────────────────────
def records_node(state: IPLAgentState) -> IPLAgentState:
    query = _resolve_retrieval_query(state)
    docs = retrieve(query, section="records")
    activated = state.get("nodes_activated", [])
    activated.append("RecordsNode")
    return {**state, "retrieved_chunks": docs, "nodes_activated": activated}


# ── NODE 8: SynthesisNode ───────────────────────────────────────────────────
def synthesis_node(state: IPLAgentState) -> IPLAgentState:
    """Combines all retrieved context and calls LLM for final answer."""
    combined_docs = (
        state.get("batting_context", [])
        + state.get("bowling_context", [])
        + state.get("h2h_context", [])
        + state.get("venue_context", [])
        + state.get("form_context", [])
        + state.get("retrieved_chunks", [])
    )
    all_docs = []
    seen = set()
    for doc in combined_docs:
        key = (
            doc.metadata.get("source", "IPL Dataset"),
            doc.metadata.get("page", ""),
            doc.page_content[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        all_docs.append(doc)

    activated = state.get("nodes_activated", [])
    activated.append("SynthesisNode")

    validation_notes = state.get("synthesised_context", "").strip()

    confidence_score, confidence_level = _compute_confidence(state, all_docs)
    source_attribution = _collect_source_attribution(all_docs)

    if not all_docs:
        return {
            **state,
            "final_answer": "Information not available in dataset.",
            "sources": [],
            "confidence": confidence_score,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "source_attribution": source_attribution,
            "nodes_activated": activated,
        }

    context_blocks = []
    for idx, doc in enumerate(all_docs[:8], start=1):
        metadata = doc.metadata or {}
        section = metadata.get("section", "unknown")
        page = metadata.get("page", "n/a")
        context_blocks.append(
            f"[Chunk {idx} | section={section} | page={page}]\n{doc.page_content}"
        )
    context_text = "\n\n".join(context_blocks)
    sources = sorted(set(
        d.metadata.get("source", "IPL Dataset") for d in all_docs
    ))

    prompt = f"""You are an expert IPL cricket analyst assistant.
Use ONLY the retrieved context below to answer the query.

Rules:
- Do not use outside IPL knowledge.
- Do not guess, infer unsupported facts, or fill missing values.
- If the required information is missing, answer exactly: "Information not available in dataset."
- If validation notes say rows were filtered, answer only from those validated rows.
- If the query has a numeric condition, every listed result must satisfy it and include the supporting value.
- For prediction questions, do not declare a winner unless the context explicitly supports the conclusion. If evidence is partial, state only the supported factors.
- Do not combine a winner claim with "Information not available in dataset."
- Prefer a compact table or bullets for lists and comparisons.

Validation notes:
{validation_notes or "None"}

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

Query: {state["user_query"]}

Answer:"""

    confidence_score, confidence_level = _compute_confidence(state, all_docs)
    source_attribution = _collect_source_attribution(all_docs)
    print("[CONFIDENCE]")
    print(f"Score: {confidence_score:.4f}")
    print(f"Level: {confidence_level}")
    print("[SOURCES]")
    print(f"Source Count: {len(source_attribution)}")
    print(
        "Source List: " + ", ".join(
            f"{item['section']}|page {item['page']}" for item in source_attribution
        )
    )

    llm = _get_llm()
    if llm is None:
        # No LLM available (no GROQ_API_KEY). Return a safe fallback summarising retrieved context.
        summary = []
        for idx, doc in enumerate(all_docs[:6], start=1):
            preview = " ".join(doc.page_content.split())[:240]
            summary.append(f"[Chunk {idx}] section={doc.metadata.get('section','n/a')} source={doc.metadata.get('source','IPL Dataset')} preview={preview}")
        fallback = "\n\n".join(summary) or "Information not available in dataset."
        return {
            **state,
            "final_answer": fallback,
            "sources": sources,
            "confidence": confidence_score,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "source_attribution": source_attribution,
            "nodes_activated": activated,
        }

    response = llm.invoke(prompt)

    return {
        **state,
        "final_answer": response.content,
        "sources": sources,
        "confidence": confidence_score,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "source_attribution": source_attribution,
        "nodes_activated": activated,
    }
