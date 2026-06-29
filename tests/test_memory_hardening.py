"""Adversarial hardening tests for the multi-turn scenario-state memory.

These actively try to break the redesigned memory: corrupting the backbone and
recovering it, driving 50 follow-ups to probe for unbounded growth, replacing
the case and continuing on the new one, mixing Chinese follow-ups, and checking
the conservative safety bias (a "what about" follow-up never silently drops the
active case). They complement ``test_memory_scenario_policy.py``.
"""

from rag_core.memory import (
    ACTION_PRESERVE,
    ACTION_REPAIR,
    ACTION_REPLACE,
    DELTA_SUMMARY_CHARS,
    MAX_CASE_DELTAS,
    ConversationMemory,
)

SCENARIO = "Funds show rapid movement through a virtual asset exchange."
NEW_CASE = (
    "A shell company with opaque ownership receives large incoming transfers "
    "and cannot identify its beneficial owner."
)


def _flag(code):
    return {"code": code, "name": code, "name_zh": code, "reason": "diagnostic"}


def _cit(chunk_id):
    return {"chunk_id": chunk_id, "source": "diagnostic", "excerpt": "evidence"}


def _seed_case(memory):
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query=SCENARIO,
        answer="a1",
        assessment="possible",
        flags=[_flag("RF-02"), _flag("RF-07")],
        citations=[_cit("c1")],
        retrieved_chunk_ids=["c1"],
        context_terms=["rapid_movement", "virtual_assets"],
    )


def _followup(memory, query, *, topics):
    memory.record_retrieval_turn(
        intent_route="retrieve_with_memory",
        user_query=query,
        answer="a",
        assessment="possible",
        flags=[],
        citations=[],
        retrieved_chunk_ids=[],
        context_terms=topics,
    )


# --- drift recovery ----------------------------------------------------------


def test_repair_scenario_restores_externally_corrupted_backbone():
    memory = ConversationMemory(session_id="repair")
    _seed_case(memory)
    # Simulate corruption: something overwrites the backbone with a fragment.
    memory.active_scenario_summary = "What about cross-border transfers?"
    assert memory.scenario_health().drift is True

    assert memory.repair_scenario() is True
    assert "rapid movement" in memory.active_scenario_summary.lower()
    assert memory.last_scenario_action == ACTION_REPAIR
    # idempotent: once healthy, repair is a no-op
    assert memory.repair_scenario() is False


def test_repair_scenario_is_noop_when_healthy():
    memory = ConversationMemory(session_id="repair-healthy")
    _seed_case(memory)
    assert memory.repair_scenario() is False
    assert memory.scenario_health().drift is False


# --- replace then continue on the new case -----------------------------------


def test_followup_after_replace_builds_on_new_case_only():
    memory = ConversationMemory(session_id="replace-then-followup")
    _seed_case(memory)
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query=NEW_CASE,
        answer="a",
        assessment="possible",
        flags=[_flag("RF-09")],
        citations=[_cit("c9")],
        retrieved_chunk_ids=["c9"],
        context_terms=["shell_company"],
    )
    assert memory.last_scenario_action == ACTION_REPLACE

    _followup(memory, "What about cross-border transfers?", topics=["cross_border"])
    composed = memory.compose_retrieval_query("What about cross-border transfers?").lower()
    assert "shell company" in composed
    # the replaced case is fully gone — no leakage of the original backbone
    assert "rapid movement" not in composed
    assert "virtual asset" not in composed


def test_consecutive_new_cases_each_replace_the_backbone():
    memory = ConversationMemory(session_id="serial-replace")
    _seed_case(memory)
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query=NEW_CASE,
        answer="a",
        assessment="possible",
        flags=[_flag("RF-09")],
        citations=[],
        retrieved_chunk_ids=[],
        context_terms=["shell_company"],
    )
    assert memory.last_scenario_action == ACTION_REPLACE

    case_c = (
        "Multiple cash deposits are structured just below the reporting "
        "threshold across many branches."
    )
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query=case_c,
        answer="a",
        assessment="possible",
        flags=[_flag("RF-01")],
        citations=[],
        retrieved_chunk_ids=[],
        context_terms=["cash_structuring"],
    )
    assert memory.last_scenario_action == ACTION_REPLACE
    summary = memory.active_scenario_summary.lower()
    assert "cash deposits" in summary or "structured" in summary
    assert "shell company" not in summary
    assert memory.active_flag_codes == ["RF-01"]


# --- bounded growth ----------------------------------------------------------


def test_compose_query_stays_bounded_across_fifty_followups():
    memory = ConversationMemory(session_id="growth")
    _seed_case(memory)
    for index in range(50):
        _followup(
            memory,
            f"What about additional risk factor number {index}?",
            topics=[],
        )
    composed = memory.compose_retrieval_query("What about yet another factor?")
    upper_bound = 500 + MAX_CASE_DELTAS * (DELTA_SUMMARY_CHARS + 1) + 300
    assert len(composed) <= upper_bound
    # the original case still anchors the composed query after 50 follow-ups
    assert "rapid movement" in composed.lower()
    assert len(memory.active_case_deltas) <= MAX_CASE_DELTAS


def test_repeated_identical_followup_does_not_duplicate_delta():
    memory = ConversationMemory(session_id="dedup")
    _seed_case(memory)
    for _ in range(3):
        _followup(memory, "What about profile mismatch?", topics=["identity_mismatch"])
    assert memory.active_case_deltas == ["profile mismatch"]


# --- Chinese follow-ups ------------------------------------------------------


def test_chinese_followup_preserves_backbone_and_distills_delta():
    memory = ConversationMemory(session_id="zh")
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query="學生帳戶由叔叔代為操作，資金入帳後立即轉出，本人無法說明用途。",
        answer="a",
        assessment="possible",
        flags=[_flag("RF-04")],
        citations=[_cit("c1")],
        retrieved_chunk_ids=["c1"],
        context_terms=["third_party", "rapid_movement"],
    )
    _followup(memory, "那跟客戶職業不符有關嗎？", topics=["identity_mismatch"])

    assert memory.last_scenario_action == ACTION_PRESERVE
    assert "學生帳戶" in memory.active_scenario_summary
    joined = " ".join(memory.active_case_deltas)
    assert "客戶職業不符" in joined
    assert not joined.startswith("那跟")


# --- conservative safety bias ------------------------------------------------


def test_followup_connector_with_new_topic_preserves_not_replaces():
    # A follow-up *phrased* as a question ("what about …") that happens to
    # introduce a new topic must never silently drop the active case: the
    # follow-up route always preserves the backbone.
    memory = ConversationMemory(session_id="safety-bias")
    _seed_case(memory)
    _followup(
        memory,
        "What about a shell company with opaque ownership?",
        topics=["shell_company"],
    )
    assert memory.last_scenario_action == ACTION_PRESERVE
    assert "rapid movement" in memory.active_scenario_summary.lower()
