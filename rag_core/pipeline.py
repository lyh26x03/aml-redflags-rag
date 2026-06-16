"""RAG pipeline with optional intent routing and structured memory.

Two modes share the same retrieval + generation core:

- **Single-turn** (default): when a request has no ``session_id`` /
  ``use_memory``, the pipeline behaves exactly as the original single-turn
  service. Memory is neither read nor written. This path is backward
  compatible; only inert memory fields are added to ``debug``.

- **Structured memory** (opt-in): when ``use_memory`` is true and a
  ``session_id`` is supplied (and ``memory_mode`` is not ``"off"``), a
  deterministic :class:`IntentRouter` chooses a route and the pipeline reads
  and updates a bounded :class:`ConversationMemory` for that session.

Retrieval math is untouched: memory is only used to *compose* a richer query
string for the ``retrieve_with_memory`` route; the BM25/dense/RRF algorithms
are unchanged.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag_core.config import Settings
from rag_core.gate import GateResult, SemanticScopeClassifier, check_scope
from rag_core.generation import RF_CATALOG, generate
from rag_core.intent_router import (
    ROUTE_ANSWER_FROM_HISTORY,
    ROUTE_ASK_CLARIFYING_QUESTION,
    ROUTE_REFUSE,
    ROUTE_RETRIEVE,
    ROUTE_RETRIEVE_WITH_MEMORY,
    IntentRouter,
    RouteDecision,
    route_family,
)
from rag_core.memory import ConversationMemory, ConversationMemoryStore
from rag_core.retrieval import RetrievalResult, Retriever
from rag_core.schemas import QueryRequest, QueryResponse, RetrievalDebug

CLARIFY_ANSWER = (
    "你的描述較為簡略，目前的資訊不足以負責任地進行 AML 紅旗判斷。"
    "請補充交易情境細節，例如資金流向、金額與頻率、客戶身分或職業、"
    "是否涉及虛擬資產或跨境匯款等，我才能依證據進行分析。 "
    "(The scenario is under-specified; please add transaction details so the "
    "analysis can be evidence-based.)"
)

REFUSE_ANSWER = (
    "這個請求超出本系統的 AML 紅旗分析範圍，無法處理。 "
    "(This request is outside the AML red-flag analysis scope.)"
)

NO_HISTORY_ANSWER = (
    "目前這個對話沒有可參考的先前分析內容。請先描述要分析的 AML 情境，"
    "我才能引用先前識別的紅旗與證據來源。 "
    "(No previous analysis is available in this session yet — describe an AML "
    "scenario first.)"
)


@dataclass
class _MemoryDebug:
    """Carrier for the additive memory/routing debug fields."""

    intent_route: Optional[str] = None
    route_reason: Optional[str] = None
    memory_used: bool = False
    memory_available: bool = False
    memory_updated: bool = False
    memory_turn_count: int = 0
    session_id: Optional[str] = None
    referenced_previous_answer: bool = False
    referenced_previous_evidence: bool = False
    active_flags: List[str] = field(default_factory=list)
    active_citation_count: int = 0


def _empty_generation_debug(llm_mode: str) -> Dict[str, Any]:
    """Generation-debug shape for routes that never call generation."""
    return {
        "requested_llm_mode": llm_mode,
        "effective_llm_mode": "mock",
        "llm_model_name": None,
        "fallback_used": False,
        "fallback_reason": None,
        "error_type": None,
        "http_status": None,
    }


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        scope_classifier: Optional[SemanticScopeClassifier] = None,
        memory_store: Optional[ConversationMemoryStore] = None,
        intent_router: Optional[IntentRouter] = None,
    ):
        self.settings = settings
        self.retriever = retriever
        self.scope_classifier = scope_classifier
        self.memory_store = memory_store or ConversationMemoryStore()
        self.intent_router = intent_router or IntentRouter()

    # --- public entry point -------------------------------------------------

    def analyze(self, request: QueryRequest) -> QueryResponse:
        requested_mode = request.retrieval_mode or self.settings.default_retrieval_mode
        requested_llm_mode = request.llm_mode or self.settings.llm_mode
        top_k = request.top_k or self.settings.default_top_k
        include_debug = self.settings.enable_debug and request.include_debug is not False

        # Optional explicit reset (only meaningful with a session id).
        if request.session_id and request.reset_memory:
            self.memory_store.reset(request.session_id)

        memory_enabled = bool(
            request.use_memory
            and request.session_id
            and (request.memory_mode or "structured") != "off"
        )

        gate = check_scope(request.query, scope_classifier=self.scope_classifier)

        existing = (
            self.memory_store.get(request.session_id) if request.session_id else None
        )
        has_memory = existing is not None and existing.has_content

        decision = self.intent_router.route(
            request.query,
            gate_allowed=gate.allowed,
            memory_enabled=memory_enabled,
            has_memory=has_memory,
        )

        ctx = _RequestContext(
            request=request,
            requested_mode=requested_mode,
            requested_llm_mode=requested_llm_mode,
            top_k=top_k,
            include_debug=include_debug,
            gate=gate,
            decision=decision,
            memory_enabled=memory_enabled,
            memory_available=has_memory,
        )

        if not memory_enabled:
            return self._single_turn(ctx)
        return self._memory_turn(ctx)

    # --- single-turn (backward compatible) ----------------------------------

    def _single_turn(self, ctx: "_RequestContext") -> QueryResponse:
        generated, retrieval, generation_debug = self._retrieve_and_generate(
            ctx.request.query, ctx
        )
        mem = _MemoryDebug(
            intent_route=ctx.decision.route,
            route_reason=ctx.decision.reason,
            session_id=ctx.request.session_id,
            memory_available=ctx.memory_available,
        )
        return self._finalize(generated, retrieval, generation_debug, ctx, mem)

    # --- structured-memory routing ------------------------------------------

    def _memory_turn(self, ctx: "_RequestContext") -> QueryResponse:
        route = ctx.decision.route
        if route == ROUTE_REFUSE:
            return self._handle_refuse(ctx)
        if route == ROUTE_ASK_CLARIFYING_QUESTION:
            return self._handle_ask_clarifying_question(ctx)
        if route == ROUTE_ANSWER_FROM_HISTORY:
            return self._handle_answer_from_history(ctx)
        if route == ROUTE_RETRIEVE_WITH_MEMORY:
            return self._handle_retrieve(ctx, use_memory_context=True)
        return self._handle_retrieve(ctx, use_memory_context=False)

    def _handle_retrieve(
        self, ctx: "_RequestContext", *, use_memory_context: bool
    ) -> QueryResponse:
        memory = self.memory_store.get_or_create(ctx.request.session_id)

        retrieval_query = ctx.request.query
        if use_memory_context and memory.active_scenario_summary:
            # Compose prior scenario + the new question. This enriches the
            # query string only; retrieval math is unchanged.
            retrieval_query = (
                f"{memory.active_scenario_summary}\n{ctx.request.query}"
            )

        generated, retrieval, generation_debug = self._retrieve_and_generate(
            retrieval_query, ctx
        )

        context_terms = sorted(
            self.intent_router.topic_detector.detect_topics(retrieval_query)
        )
        memory.resolve_clarifications()
        memory.record_retrieval_turn(
            intent_route=ctx.decision.route,
            user_query=ctx.request.query,
            answer=generated["answer"],
            assessment=generated["assessment"],
            flags=generated["identified_flags"],
            citations=generated["citations"],
            retrieved_chunk_ids=retrieval.chunk_ids,
            context_terms=context_terms,
        )

        mem = self._memory_debug(
            ctx,
            memory,
            memory_used=use_memory_context,
            memory_updated=True,
            referenced_previous_answer=use_memory_context,
            referenced_previous_evidence=(
                use_memory_context and bool(memory.active_citations)
            ),
        )
        return self._finalize(generated, retrieval, generation_debug, ctx, mem)

    def _handle_answer_from_history(self, ctx: "_RequestContext") -> QueryResponse:
        memory = self.memory_store.get_or_create(ctx.request.session_id)
        retrieval = RetrievalResult(
            requested_mode=ctx.requested_mode, effective_mode="none"
        )
        generation_debug = _empty_generation_debug(ctx.requested_llm_mode)

        if not memory.has_content:
            generated = {
                "answer": NO_HISTORY_ANSWER,
                "assessment": "unlikely",
                "identified_flags": [],
                "citations": [],
                "refusal": {"refused": False, "reason": None},
                "parse_success": None,
            }
            mem = self._memory_debug(
                ctx,
                memory,
                memory_used=False,
                memory_updated=False,
                memory_available=False,
                referenced_previous_answer=False,
                referenced_previous_evidence=False,
            )
            return self._finalize(generated, retrieval, generation_debug, ctx, mem)

        flags = self._flags_from_memory(memory)
        citations = [c.to_dict() for c in memory.active_citations]
        wants_evidence = ctx.decision.referenced_evidence
        answer = self._compose_history_answer(memory, wants_evidence)
        assessment = memory.last_assessment or ("possible" if flags else "unlikely")

        generated = {
            "answer": answer,
            "assessment": assessment,
            "identified_flags": flags,
            "citations": citations,
            "refusal": {"refused": False, "reason": None},
            "parse_success": None,
        }
        memory.record_history_turn(
            user_query=ctx.request.query,
            answer=answer,
            assessment=assessment,
        )
        mem = self._memory_debug(
            ctx,
            memory,
            memory_used=True,
            memory_updated=True,
            referenced_previous_answer=True,
            referenced_previous_evidence=bool(citations),
        )
        return self._finalize(generated, retrieval, generation_debug, ctx, mem)

    def _handle_ask_clarifying_question(
        self, ctx: "_RequestContext"
    ) -> QueryResponse:
        memory = self.memory_store.get_or_create(ctx.request.session_id)
        memory.record_clarify_turn(
            user_query=ctx.request.query, clarification=CLARIFY_ANSWER
        )
        generated = {
            "answer": CLARIFY_ANSWER,
            "assessment": "unlikely",
            "identified_flags": [],
            "citations": [],
            "refusal": {"refused": False, "reason": None},
            "parse_success": None,
        }
        retrieval = RetrievalResult(
            requested_mode=ctx.requested_mode, effective_mode="none"
        )
        mem = self._memory_debug(
            ctx,
            memory,
            memory_used=False,
            memory_updated=True,
        )
        return self._finalize(
            generated,
            retrieval,
            _empty_generation_debug(ctx.requested_llm_mode),
            ctx,
            mem,
        )

    def _handle_refuse(self, ctx: "_RequestContext") -> QueryResponse:
        # Refusals never touch memory — out-of-scope requests must not pollute
        # the active AML scenario state.
        reason = ctx.gate.reason_message or REFUSE_ANSWER
        generated = {
            "answer": reason,
            "assessment": "refuse",
            "identified_flags": [],
            "citations": [],
            "refusal": {"refused": True, "reason": reason},
            "parse_success": None,
        }
        retrieval = RetrievalResult(
            requested_mode=ctx.requested_mode, effective_mode="none"
        )
        existing = self.memory_store.get(ctx.request.session_id)
        mem = _MemoryDebug(
            intent_route=ctx.decision.route,
            route_reason=ctx.decision.reason,
            session_id=ctx.request.session_id,
            memory_used=False,
            memory_updated=False,
            memory_available=ctx.memory_available,
            memory_turn_count=existing.turn_count if existing else 0,
            active_flags=list(existing.active_flag_codes) if existing else [],
            active_citation_count=len(existing.active_citations) if existing else 0,
        )
        return self._finalize(
            generated,
            retrieval,
            _empty_generation_debug(ctx.requested_llm_mode),
            ctx,
            mem,
        )

    # --- shared retrieval + generation core ---------------------------------

    def _retrieve_and_generate(
        self, query: str, ctx: "_RequestContext"
    ):
        gate = ctx.gate
        if gate.allowed:
            retrieval = self.retriever.retrieve(
                query,
                top_k=ctx.top_k,
                requested_mode=ctx.requested_mode,
            )
        else:
            retrieval = RetrievalResult(
                requested_mode=ctx.requested_mode,
                effective_mode="none",
            )

        generated = generate(
            query=query,
            chunks=retrieval.contexts,
            llm_mode=ctx.requested_llm_mode,
            model_name=self.settings.model_name,
            gemini_api_key=self.settings.gemini_api_key,
            groq_api_key=self.settings.groq_api_key,
            llm_timeout_seconds=self.settings.llm_timeout_seconds,
            ollama_base_url=self.settings.ollama_base_url,
            ollama_model=self.settings.ollama_model,
            gate_allowed=gate.allowed,
            gate_reason=gate.reason_message or gate.reason_code,
        )
        generation_debug = generated.pop("_generation_debug")
        return generated, retrieval, generation_debug

    # --- response + debug assembly ------------------------------------------

    def _finalize(
        self,
        generated: Dict[str, Any],
        retrieval: RetrievalResult,
        generation_debug: Dict[str, Any],
        ctx: "_RequestContext",
        mem: _MemoryDebug,
    ) -> QueryResponse:
        if not ctx.include_debug:
            return QueryResponse(**generated, debug=None)

        fallback_reasons = [
            reason
            for reason in (
                retrieval.fallback_reason,
                generation_debug["fallback_reason"],
            )
            if reason
        ]
        debug = RetrievalDebug(
            retrieval_mode=retrieval.effective_mode,
            requested_mode=ctx.requested_mode,
            top_k=ctx.top_k,
            dense_used=retrieval.dense_used,
            bm25_used=retrieval.bm25_used,
            rrf_used=retrieval.rrf_used,
            gate_decision=ctx.gate.decision_label,
            llm_mode=generation_debug["effective_llm_mode"],
            llm_model_name=generation_debug["llm_model_name"],
            fallback_used=(
                retrieval.fallback_used or generation_debug["fallback_used"]
            ),
            fallback_reason="; ".join(fallback_reasons) or None,
            error_type=generation_debug["error_type"],
            http_status=generation_debug["http_status"],
            parse_success=generated.get("parse_success"),
            retrieved_chunk_ids=retrieval.chunk_ids,
            intent_route=mem.intent_route,
            route_family=route_family(mem.intent_route),
            route_reason=mem.route_reason,
            memory_used=mem.memory_used,
            memory_available=mem.memory_available,
            memory_updated=mem.memory_updated,
            memory_turn_count=mem.memory_turn_count,
            session_id=mem.session_id,
            referenced_previous_answer=mem.referenced_previous_answer,
            referenced_previous_evidence=mem.referenced_previous_evidence,
            active_flags=mem.active_flags,
            active_citation_count=mem.active_citation_count,
        )
        return QueryResponse(**generated, debug=debug)

    # --- memory helpers -----------------------------------------------------

    def _memory_debug(
        self,
        ctx: "_RequestContext",
        memory: ConversationMemory,
        *,
        memory_used: bool,
        memory_updated: bool,
        memory_available: Optional[bool] = None,
        referenced_previous_answer: bool = False,
        referenced_previous_evidence: bool = False,
    ) -> _MemoryDebug:
        return _MemoryDebug(
            intent_route=ctx.decision.route,
            route_reason=ctx.decision.reason,
            memory_used=memory_used,
            memory_updated=memory_updated,
            memory_available=(
                ctx.memory_available if memory_available is None else memory_available
            ),
            memory_turn_count=memory.turn_count,
            session_id=ctx.request.session_id,
            referenced_previous_answer=referenced_previous_answer,
            referenced_previous_evidence=referenced_previous_evidence,
            active_flags=list(memory.active_flag_codes),
            active_citation_count=len(memory.active_citations),
        )

    @staticmethod
    def _flags_from_memory(memory: ConversationMemory) -> List[Dict[str, Any]]:
        flags: List[Dict[str, Any]] = []
        for flag in memory.active_flags:
            code = str(flag.get("code", ""))
            catalog = RF_CATALOG.get(code, {})
            flags.append(
                {
                    "code": code,
                    "name": flag.get("name") or catalog.get("name", code),
                    "name_zh": flag.get("name_zh") or catalog.get("name_zh"),
                    "reason": "已於先前的情境分析中識別出此紅旗。",
                }
            )
        return flags

    @staticmethod
    def _compose_history_answer(
        memory: ConversationMemory, wants_evidence: bool
    ) -> str:
        parts: List[str] = []
        if memory.active_flags:
            flag_text = "、".join(
                f"{flag.get('code')} {flag.get('name')}"
                for flag in memory.active_flags
            )
            parts.append(f"先前分析中識別出的紅旗為：{flag_text}。")
        else:
            parts.append("先前的分析尚未識別出具體紅旗。")

        if memory.last_assessment:
            parts.append(f"當時的整體判斷為 {memory.last_assessment}。")

        if wants_evidence or memory.active_citations:
            if memory.active_citations:
                source_text = "；".join(
                    f"{c.source} ({c.chunk_id})"
                    for c in memory.active_citations
                )
                parts.append(f"引用的證據來源包括：{source_text}。")
            else:
                parts.append("先前的分析沒有附帶可引用的證據來源。")

        parts.append(
            "(Recalled from this session's structured memory; no new retrieval "
            "was performed.)"
        )
        return " ".join(parts)


@dataclass
class _RequestContext:
    """Resolved per-request parameters shared across route handlers."""

    request: QueryRequest
    requested_mode: str
    requested_llm_mode: str
    top_k: int
    include_debug: bool
    gate: GateResult
    decision: RouteDecision
    memory_enabled: bool
    memory_available: bool
