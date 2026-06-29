"""Pydantic request/response models. Field names follow docs/demo_spec.md §6.

Fields beyond the spec (`name_zh`, `requested_mode`, `fallback_reason`,
`message` on SourcesResponse) are additive only — spec fields are never
removed or renamed.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

RetrievalMode = Literal["hybrid", "dense", "bm25"]
LlmMode = Literal["mock", "gemini", "gemma", "groq", "ollama"]
Assessment = Literal["confirmed", "possible", "unlikely", "refuse"]
GateDecisionLiteral = Literal["allow", "refuse"]
MemoryMode = Literal["off", "structured"]
IntentRoute = Literal[
    "retrieve",
    "refuse",
    "ask_clarifying_question",
    "answer_from_history",
    "retrieve_with_memory",
]
# Three reviewer-facing outcomes that the five fine-grained routes collapse onto.
RouteFamily = Literal["retrieve", "refuse", "no_retrieval_response"]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    retrieval_mode: Optional[RetrievalMode] = None
    llm_mode: Optional[LlmMode] = None
    include_debug: Optional[bool] = None
    # --- structured conversation memory (additive; all optional) ---
    # Existing single-turn clients omit these and keep their exact behavior.
    session_id: Optional[str] = Field(default=None, max_length=200)
    use_memory: bool = False
    memory_mode: Optional[MemoryMode] = None
    reset_memory: Optional[bool] = None


class Citation(BaseModel):
    chunk_id: str
    source: str
    excerpt: str


class IdentifiedFlag(BaseModel):
    code: str
    name: str
    reason: str
    name_zh: Optional[str] = None


class RefusalInfo(BaseModel):
    refused: bool = False
    reason: Optional[str] = None


class RetrievalDebug(BaseModel):
    retrieval_mode: str  # effective mode actually used
    top_k: int
    dense_used: bool
    bm25_used: bool
    rrf_used: bool
    gate_decision: GateDecisionLiteral
    llm_mode: str
    fallback_used: bool
    retrieved_chunk_ids: List[str]
    requested_mode: Optional[str] = None
    llm_model_name: Optional[str] = None
    fallback_reason: Optional[str] = None
    error_type: Optional[str] = None
    http_status: Optional[int] = None
    parse_success: Optional[bool] = None
    # --- intent routing + structured conversation memory (additive) ---
    # Present for all requests; inert defaults keep single-turn debug stable.
    # ``route_family`` is the high-level (three-outcome) view of ``intent_route``.
    intent_route: Optional[IntentRoute] = None
    route_family: Optional[RouteFamily] = None
    route_reason: Optional[str] = None
    memory_used: bool = False
    memory_available: bool = False
    memory_updated: bool = False
    memory_turn_count: int = 0
    session_id: Optional[str] = None
    referenced_previous_answer: bool = False
    referenced_previous_evidence: bool = False
    active_flags: List[str] = Field(default_factory=list)
    active_citation_count: int = 0
    # Scenario-state policy audit: which action the memory took on the case
    # backbone this turn (seed/preserve/replace/...) and how many follow-up
    # deltas are currently accumulated.
    scenario_update_action: Optional[str] = None
    case_delta_count: int = 0


class QueryResponse(BaseModel):
    answer: str
    assessment: Assessment
    identified_flags: List[IdentifiedFlag] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    refusal: RefusalInfo = Field(default_factory=RefusalInfo)
    parse_success: Optional[bool] = None
    debug: Optional[RetrievalDebug] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str = "aml-redflags-rag-api"
    corpus_profile: str = "sample"
    artifacts_loaded: bool
    llm_mode: str
    model_name: str
    index_version: Optional[str] = None
    chunk_count: int = 0
    source_count: int = 0
    source_names: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class SourceSummary(BaseModel):
    source_name: str
    language: str
    layer: str
    chunk_count: int


class SourcesResponse(BaseModel):
    corpus_profile: str = "sample"
    index_version: Optional[str] = None
    chunk_count: int = 0
    total_chunks: int = 0
    source_count: int = 0
    source_names: List[str] = Field(default_factory=list)
    sources: List[SourceSummary] = Field(default_factory=list)
    message: Optional[str] = None


# --- structured conversation memory inspection (demo/debug only) ---


class MemoryCitationView(BaseModel):
    chunk_id: str
    source: str
    excerpt: str


class MemoryTurnView(BaseModel):
    turn_index: int
    intent_route: str
    user_query: str
    assessment: Optional[str] = None
    answer_summary: str = ""
    flag_codes: List[str] = Field(default_factory=list)
    citation_chunk_ids: List[str] = Field(default_factory=list)
    created_at: str


class SessionMemoryResponse(BaseModel):
    """Read-only snapshot of one session's bounded structured memory."""

    session_id: str
    turn_count: int = 0
    active_scenario_summary: str = ""
    active_case_deltas: List[str] = Field(default_factory=list)
    case_seed_text: str = ""
    scenario_origin_turn: int = 0
    last_scenario_action: str = ""
    scenario_health: Optional[dict] = None
    active_entities_or_context_terms: List[str] = Field(default_factory=list)
    active_flags: List[dict] = Field(default_factory=list)
    active_citations: List[MemoryCitationView] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    last_assessment: Optional[str] = None
    last_answer_summary: str = ""
    unresolved_questions: List[str] = Field(default_factory=list)
    recent_turns: List[MemoryTurnView] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MemoryDeleteResponse(BaseModel):
    session_id: str
    deleted: bool
    message: str
