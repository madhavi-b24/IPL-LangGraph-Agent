from typing import TypedDict, List
from langchain_core.documents import Document


class IPLAgentState(TypedDict):
    # Input
    user_query: str
    query_type: str                  # 'batting'|'bowling'|'h2h'|'venue'|'form'|'records'|'prediction'|'dream11'
    entities: List[str]              # extracted player/team names

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
    confidence: float
    nodes_activated: List[str]       # tracks which nodes ran (for UI display)