from typing import Any, TypedDict, List,Dict,Optional,NotRequired
from langchain_core.documents import Document


class IPLAgentState(TypedDict):
    # Input
    user_query: str
    rewritten_query:str
    query_type: str                  # 'batting'|'bowling'|'h2h'|'venue'|'form'|'records'|'prediction'|'dream11'
    entities: List[str]              # extracted player/team names
    selected_tool: str
    tool_confidence: float
    query_confidence: float
    selected_agent: NotRequired[str]
    agent_confidence: NotRequired[float]
    chat_history: NotRequired[List[dict]]
    conversation_entities: NotRequired[dict]
    conversation_summary: NotRequired[str]

    # Retrieved context per node
    batting_context: List[Document]
    bowling_context: List[Document]
    h2h_context: List[Document]
    venue_context: List[Document]
    form_context: List[Document]
    retrieved_chunks: List[Document]  # used by RecordsNode

    # Intermediate
    synthesised_context: str

    # Final output
    final_answer: str
    sources: List[str]
    citations: NotRequired[List[dict]]
    confidence: float
    confidence_score: float
    confidence_level: str
    # Verification metadata
    verification_score: float
    verification_passed: bool
    verification_reason: str
    # Prediction calibration
    prediction_confidence: Optional[float]
    evidence_strength: str
    supporting_sources: int
    source_attribution: List[dict]
    hallucination_score: NotRequired[float]
    hallucination_flag: NotRequired[bool]
    hallucination_reason: NotRequired[str]
    evaluation: NotRequired[dict]
    overall_quality_score: NotRequired[float]
    quality_level: NotRequired[str]
    agent_metadata_filters: NotRequired[dict]
    agent_retrieval_strategy: NotRequired[Any]
    nodes_activated: List[str]       # tracks which nodes ran (for UI display)
