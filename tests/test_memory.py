"""Unit tests for structured conversation memory state, store, and router."""

from rag_core.intent_router import (
    FAMILY_NO_RETRIEVAL,
    FAMILY_REFUSE,
    FAMILY_RETRIEVE,
    ROUTE_ANSWER_FROM_HISTORY,
    ROUTE_ASK_CLARIFYING_QUESTION,
    ROUTE_REFUSE,
    ROUTE_RETRIEVE,
    ROUTE_RETRIEVE_WITH_MEMORY,
    IntentRouter,
    route_family,
)
from rag_core.memory import (
    MAX_ACTIVE_CITATIONS,
    MAX_RECENT_TURNS,
    ConversationMemory,
    ConversationMemoryStore,
)


def _flag(code, name="Name", name_zh="名稱"):
    return {"code": code, "name": name, "name_zh": name_zh, "reason": "because"}


def _citation(chunk_id, source="Src", excerpt="text"):
    return {"chunk_id": chunk_id, "source": source, "excerpt": excerpt}


# --- state: bounds, dedup, truncation ---------------------------------------


def test_recent_turns_are_bounded():
    memory = ConversationMemory(session_id="s")
    for index in range(MAX_RECENT_TURNS + 5):
        memory.record_retrieval_turn(
            intent_route="retrieve",
            user_query=f"scenario {index}",
            answer="ans",
            assessment="possible",
            flags=[_flag("RF-02")],
            citations=[_citation(f"c{index}")],
            retrieved_chunk_ids=[f"c{index}"],
            context_terms=["rapid_movement"],
        )
    assert len(memory.recent_turns) == MAX_RECENT_TURNS
    # turn_count keeps counting even though raw turns are trimmed
    assert memory.turn_count == MAX_RECENT_TURNS + 5
    # most recent turn is retained
    assert memory.recent_turns[-1].turn_index == MAX_RECENT_TURNS + 5


def test_active_flags_are_deduplicated_by_code():
    memory = ConversationMemory(session_id="s")
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query="scenario",
        answer="ans",
        assessment="possible",
        flags=[_flag("RF-02"), _flag("RF-02"), _flag("RF-07")],
        citations=[],
        retrieved_chunk_ids=[],
        context_terms=[],
    )
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query="scenario 2",
        answer="ans",
        assessment="possible",
        flags=[_flag("RF-02"), _flag("RF-06")],
        citations=[],
        retrieved_chunk_ids=[],
        context_terms=[],
    )
    assert memory.active_flag_codes == ["RF-02", "RF-07", "RF-06"]


def test_active_citations_are_bounded_and_deduplicated():
    memory = ConversationMemory(session_id="s")
    # add more than the bound, with a duplicate chunk id
    citations = [_citation(f"c{i}") for i in range(MAX_ACTIVE_CITATIONS + 4)]
    citations.append(_citation("c0", source="Updated"))
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query="scenario",
        answer="ans",
        assessment="possible",
        flags=[],
        citations=citations,
        retrieved_chunk_ids=[],
        context_terms=[],
    )
    assert len(memory.active_citations) == MAX_ACTIVE_CITATIONS
    chunk_ids = [c.chunk_id for c in memory.active_citations]
    assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates


def test_citation_excerpt_is_truncated():
    memory = ConversationMemory(session_id="s")
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query="scenario",
        answer="ans",
        assessment="possible",
        flags=[],
        citations=[_citation("c0", excerpt="x" * 1000)],
        retrieved_chunk_ids=[],
        context_terms=[],
    )
    assert len(memory.active_citations[0].excerpt) <= 200


def test_retrieved_chunk_ids_dedup_keep_recent():
    memory = ConversationMemory(session_id="s")
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query="s1",
        answer="a",
        assessment="possible",
        flags=[],
        citations=[],
        retrieved_chunk_ids=["c1", "c2"],
        context_terms=[],
    )
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query="s2",
        answer="a",
        assessment="possible",
        flags=[],
        citations=[],
        retrieved_chunk_ids=["c2", "c3"],
        context_terms=[],
    )
    # c2 re-appears most recently; order preserved without duplicates
    assert memory.retrieved_chunk_ids == ["c1", "c2", "c3"]


def test_clarify_turn_stores_unresolved_without_flags():
    memory = ConversationMemory(session_id="s")
    memory.record_clarify_turn(user_query="這樣有沒有問題？", clarification="please specify")
    assert memory.unresolved_questions == ["這樣有沒有問題？"]
    assert memory.active_flags == []
    assert memory.active_scenario_summary == ""
    # the turn is recorded under the renamed ask_clarifying_question route
    assert memory.recent_turns[-1].intent_route == "ask_clarifying_question"


def test_to_dict_is_json_serializable_shape():
    memory = ConversationMemory(session_id="s")
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query="scenario",
        answer="ans",
        assessment="possible",
        flags=[_flag("RF-02")],
        citations=[_citation("c0")],
        retrieved_chunk_ids=["c0"],
        context_terms=["rapid_movement"],
    )
    snapshot = memory.to_dict()
    assert snapshot["session_id"] == "s"
    assert snapshot["turn_count"] == 1
    assert snapshot["active_flags"][0]["code"] == "RF-02"
    assert snapshot["active_citations"][0]["chunk_id"] == "c0"
    assert snapshot["recent_turns"][0]["intent_route"] == "retrieve"


# --- store ------------------------------------------------------------------


def test_store_get_or_create_and_reset():
    store = ConversationMemoryStore()
    assert store.get("s") is None
    memory = store.get_or_create("s")
    assert store.get("s") is memory
    assert store.exists("s") is True
    assert store.reset("s") is True
    assert store.reset("s") is False
    assert store.get("s") is None


def test_store_evicts_least_recently_updated():
    store = ConversationMemoryStore(max_sessions=2)
    a = store.get_or_create("a")
    a.record_clarify_turn(user_query="q", clarification="c")  # touches updated_at
    b = store.get_or_create("b")
    b.record_clarify_turn(user_query="q", clarification="c")
    store.get_or_create("c")  # over the cap -> evict oldest ("a")
    assert store.exists("a") is False
    assert store.exists("b") is True
    assert store.exists("c") is True


def test_store_snapshot_returns_none_for_missing():
    store = ConversationMemoryStore()
    assert store.snapshot("missing") is None


# --- intent router ----------------------------------------------------------


def _router():
    return IntentRouter()


def test_router_single_turn_only_retrieve_or_refuse():
    router = _router()
    assert (
        router.route(
            "剛剛那個風險", gate_allowed=True, memory_enabled=False, has_memory=True
        ).route
        == ROUTE_RETRIEVE
    )
    assert (
        router.route(
            "anything", gate_allowed=False, memory_enabled=False, has_memory=False
        ).route
        == ROUTE_REFUSE
    )


def test_router_history_reference_routes_to_answer_from_history():
    router = _router()
    decision = router.route(
        "剛剛那個風險可以再說明嗎？",
        gate_allowed=True,
        memory_enabled=True,
        has_memory=True,
    )
    assert decision.route == ROUTE_ANSWER_FROM_HISTORY
    assert decision.referenced_history is True


def test_router_citation_question_flags_evidence_reference():
    router = _router()
    decision = router.route(
        "剛剛引用的是哪些來源？",
        gate_allowed=True,
        memory_enabled=True,
        has_memory=True,
    )
    assert decision.route == ROUTE_ANSWER_FROM_HISTORY
    assert decision.referenced_evidence is True


def test_router_followup_with_memory_routes_to_retrieve_with_memory():
    router = _router()
    decision = router.route(
        "那跟客戶職業不符有關嗎？",
        gate_allowed=True,
        memory_enabled=True,
        has_memory=True,
    )
    assert decision.route == ROUTE_RETRIEVE_WITH_MEMORY


def test_router_followup_without_memory_does_not_use_memory_route():
    router = _router()
    decision = router.route(
        "那跟客戶職業不符有關嗎？",
        gate_allowed=True,
        memory_enabled=True,
        has_memory=False,
    )
    # has a detected AML topic (identity_mismatch) -> plain retrieve, not
    # ask_clarifying_question
    assert decision.route == ROUTE_RETRIEVE


def test_router_vague_first_turn_routes_to_ask_clarifying_question():
    router = _router()
    decision = router.route(
        "這樣有沒有問題？",
        gate_allowed=True,
        memory_enabled=True,
        has_memory=False,
    )
    assert decision.route == ROUTE_ASK_CLARIFYING_QUESTION


def test_router_out_of_scope_routes_to_refuse():
    router = _router()
    decision = router.route(
        "幫我推薦晚餐",
        gate_allowed=True,
        memory_enabled=True,
        has_memory=True,
    )
    assert decision.route == ROUTE_REFUSE
    assert decision.reason == "router_out_of_scope"


def test_router_normal_aml_query_routes_to_retrieve():
    router = _router()
    decision = router.route(
        "Funds show rapid movement through a virtual asset exchange.",
        gate_allowed=True,
        memory_enabled=True,
        has_memory=False,
    )
    assert decision.route == ROUTE_RETRIEVE


# --- route families (the three reviewer-facing outcomes) --------------------


def test_route_family_collapses_five_routes_onto_three_outcomes():
    # retrieval outcomes
    assert route_family(ROUTE_RETRIEVE) == FAMILY_RETRIEVE
    assert route_family(ROUTE_RETRIEVE_WITH_MEMORY) == FAMILY_RETRIEVE
    # refusal outcome
    assert route_family(ROUTE_REFUSE) == FAMILY_REFUSE
    # no-retrieval (deterministic) outcomes
    assert route_family(ROUTE_ANSWER_FROM_HISTORY) == FAMILY_NO_RETRIEVAL
    assert route_family(ROUTE_ASK_CLARIFYING_QUESTION) == FAMILY_NO_RETRIEVAL


def test_route_family_is_none_only_for_none():
    assert route_family(None) is None
    # unknown labels default to the safe single-turn retrieve family
    assert route_family("something_new") == FAMILY_RETRIEVE
