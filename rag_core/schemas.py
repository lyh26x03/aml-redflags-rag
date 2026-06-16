"""Pydantic request/response models. Field names follow docs/demo_spec.md §6.

Fields beyond the spec (`name_zh`, `requested_mode`, `fallback_reason`,
`message` on SourcesResponse) are additive only — spec fields are never
removed or renamed.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

RetrievalMode = Literal["hybrid", "dense", "bm25"]
LlmMode = Literal["mock", "gemini", "gemma", "groq"]
Assessment = Literal["confirmed", "possible", "unlikely", "refuse"]
GateDecisionLiteral = Literal["allow", "refuse"]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    retrieval_mode: Optional[RetrievalMode] = None
    llm_mode: Optional[LlmMode] = None
    include_debug: Optional[bool] = None


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
