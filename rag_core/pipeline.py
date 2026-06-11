"""Single-turn RAG pipeline adapted from notebook ``analyze_scenario``."""

from typing import Optional

from rag_core.config import Settings
from rag_core.gate import SemanticScopeClassifier, check_scope
from rag_core.generation import generate
from rag_core.retrieval import RetrievalResult, Retriever
from rag_core.schemas import QueryRequest, QueryResponse, RetrievalDebug


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        scope_classifier: Optional[SemanticScopeClassifier] = None,
    ):
        self.settings = settings
        self.retriever = retriever
        self.scope_classifier = scope_classifier

    def analyze(self, request: QueryRequest) -> QueryResponse:
        requested_mode = request.retrieval_mode or self.settings.default_retrieval_mode
        requested_llm_mode = request.llm_mode or self.settings.llm_mode
        top_k = request.top_k or self.settings.default_top_k
        include_debug = self.settings.enable_debug and request.include_debug is not False

        gate = check_scope(request.query, scope_classifier=self.scope_classifier)
        if gate.allowed:
            retrieval = self.retriever.retrieve(
                request.query,
                top_k=top_k,
                requested_mode=requested_mode,
            )
        else:
            retrieval = RetrievalResult(
                requested_mode=requested_mode,
                effective_mode="none",
            )

        generated = generate(
            query=request.query,
            chunks=retrieval.contexts,
            llm_mode=requested_llm_mode,
            model_name=self.settings.model_name,
            gemini_api_key=self.settings.gemini_api_key,
            groq_api_key=self.settings.groq_api_key,
            gate_allowed=gate.allowed,
            gate_reason=gate.reason_message or gate.reason_code,
        )
        generation_debug = generated.pop("_generation_debug")

        fallback_reasons = [
            reason
            for reason in (
                retrieval.fallback_reason,
                generation_debug["fallback_reason"],
            )
            if reason
        ]
        debug = None
        if include_debug:
            debug = RetrievalDebug(
                retrieval_mode=retrieval.effective_mode,
                requested_mode=requested_mode,
                top_k=top_k,
                dense_used=retrieval.dense_used,
                bm25_used=retrieval.bm25_used,
                rrf_used=retrieval.rrf_used,
                gate_decision=gate.decision_label,
                llm_mode=generation_debug["effective_llm_mode"],
                fallback_used=(
                    retrieval.fallback_used or generation_debug["fallback_used"]
                ),
                fallback_reason="; ".join(fallback_reasons) or None,
                retrieved_chunk_ids=retrieval.chunk_ids,
            )

        return QueryResponse(**generated, debug=debug)
