"""Hardening tests for the redesigned multi-turn scenario-state memory.

These go beyond the original third-turn regression in
``test_active_scenario_summary_overwrite.py`` and exercise the redesign across
four layers:

1. the pure scenario-update **policy** (``decide_scenario_update``),
2. delta **distillation** and the **drift detector**,
3. **state**-level multi-turn behaviour (backbone vs deltas), and
4. **pipeline / API** behaviour for the six requirement scenarios, including
   fourth-turn drift, new-case replacement, history recall, and refusal.

The guiding invariant: ``active_scenario_summary`` is the *stable case
backbone*. A short follow-up may add a bounded delta but must never overwrite
the backbone; only a genuinely new standalone case replaces it.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from rag_core.config import Settings
from rag_core.intent_router import (
    ROUTE_RETRIEVE,
    ROUTE_RETRIEVE_WITH_MEMORY,
    IntentRouter,
)
from rag_core.memory import (
    ACTION_NOOP,
    ACTION_PRESERVE,
    ACTION_REPLACE,
    ACTION_SEED,
    MAX_CASE_DELTAS,
    ConversationMemory,
    decide_scenario_update,
    detect_scenario_drift,
    distill_delta,
)
from rag_core.pipeline import RAGPipeline
from rag_core.retrieval import RetrievalResult
from rag_core.schemas import QueryRequest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "index"

SCENARIO = "Funds show rapid movement through a virtual asset exchange."
FOLLOWUP_PROFILE = "What about profile mismatch?"
FOLLOWUP_CROSS_BORDER = "What about cross-border transfers?"
COMBINED_QUESTION = "What are the combined red flags?"
NEW_CASE = (
    "A shell company with opaque ownership receives large incoming transfers "
    "and cannot identify its beneficial owner."
)


# --- helpers ----------------------------------------------------------------


def _flag(code):
    return {"code": code, "name": code, "name_zh": code, "reason": "diagnostic"}


def _cit(chunk_id):
    return {"chunk_id": chunk_id, "source": "diagnostic", "excerpt": "evidence"}


def _settings():
    return Settings(
        artifact_dir=str(ARTIFACT_DIR),
        llm_mode="mock",
        model_name="mock-local",
        enable_debug=True,
    )


def _q(client, query, session_id):
    return client.post(
        "/query",
        json={
            "query": query,
            "retrieval_mode": "bm25",
            "llm_mode": "mock",
            "include_debug": True,
            "session_id": session_id,
            "use_memory": True,
        },
    )


class CapturingRetriever:
    """Records every query string the pipeline passes to retrieval."""

    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k=5, requested_mode="bm25"):
        self.queries.append(query)
        return RetrievalResult(
            contexts=[
                {
                    "chunk_id": "diagnostic-all",
                    "source": "diagnostic",
                    "page": 1,
                    "text": (
                        "rapid movement virtual asset exchange profile mismatch "
                        "cross-border transfers shell company beneficial owner"
                    ),
                    "score": 1.0,
                    "doc_category": "aml",
                    "doc_type": "diagnostic",
                    "explanation_style": "neutral",
                    "related_flags": ["RF-02", "RF-05", "RF-06", "RF-07", "RF-09"],
                }
            ],
            requested_mode=requested_mode,
            effective_mode="bm25",
            bm25_used=True,
        )


def _seed_case(memory):
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query=SCENARIO,
        answer="turn 1 answer",
        assessment="possible",
        flags=[_flag("RF-02"), _flag("RF-07")],
        citations=[_cit("c1")],
        retrieved_chunk_ids=["c1"],
        context_terms=["rapid_movement", "virtual_assets"],
    )


# --- 1. policy unit tests ----------------------------------------------------


def test_policy_seeds_first_case():
    decision = decide_scenario_update(
        current_query=SCENARIO,
        route=ROUTE_RETRIEVE,
        has_backbone=False,
        new_topics={"rapid_movement", "virtual_assets"},
        existing_topics=set(),
    )
    assert decision.action == ACTION_SEED


def test_policy_preserves_on_memory_followup():
    decision = decide_scenario_update(
        current_query=FOLLOWUP_PROFILE,
        route=ROUTE_RETRIEVE_WITH_MEMORY,
        has_backbone=True,
        new_topics={"identity_mismatch"},
        existing_topics={"rapid_movement", "virtual_assets"},
    )
    assert decision.action == ACTION_PRESERVE
    assert decision.composes_case_context is True


def test_policy_preserves_on_additive_plain_retrieve():
    decision = decide_scenario_update(
        current_query="The customer also structures cash deposits below the threshold.",
        route=ROUTE_RETRIEVE,
        has_backbone=True,
        new_topics={"cash_structuring"},
        existing_topics={"rapid_movement", "virtual_assets"},
    )
    assert decision.action == ACTION_PRESERVE
    assert decision.reason == "additive_refinement"


def test_policy_replaces_on_new_standalone_case():
    decision = decide_scenario_update(
        current_query=NEW_CASE,
        route=ROUTE_RETRIEVE,
        has_backbone=True,
        new_topics={"shell_company"},
        existing_topics={"rapid_movement", "virtual_assets"},
    )
    assert decision.action == ACTION_REPLACE


def test_policy_preserves_short_topicless_whole_case_question():
    # A whole-case question carries no new topic and is short: it must not be
    # mistaken for a new standalone case.
    decision = decide_scenario_update(
        current_query=COMBINED_QUESTION,
        route=ROUTE_RETRIEVE,
        has_backbone=True,
        new_topics=set(),
        existing_topics={"rapid_movement", "virtual_assets"},
    )
    assert decision.action == ACTION_PRESERVE


def test_policy_noop_on_empty_query():
    decision = decide_scenario_update(
        current_query="   ",
        route=ROUTE_RETRIEVE,
        has_backbone=True,
        new_topics=set(),
        existing_topics={"rapid_movement"},
    )
    assert decision.action == ACTION_NOOP


def test_policy_scorer_seam_overrides_default_rule():
    # An injected scorer can force a replace even with overlapping topics ...
    forced_new = decide_scenario_update(
        current_query="ambiguous restatement",
        route=ROUTE_RETRIEVE,
        has_backbone=True,
        new_topics={"rapid_movement"},
        existing_topics={"rapid_movement"},
        new_case_scorer=lambda **_: 1.0,
    )
    assert forced_new.action == ACTION_REPLACE

    # ... or veto a replace the default rule would have made (disjoint topics).
    forced_keep = decide_scenario_update(
        current_query=NEW_CASE,
        route=ROUTE_RETRIEVE,
        has_backbone=True,
        new_topics={"shell_company"},
        existing_topics={"rapid_movement"},
        new_case_scorer=lambda **_: 0.0,
    )
    assert forced_keep.action == ACTION_PRESERVE


# --- 2. distillation + drift detector ---------------------------------------


def test_distill_delta_strips_leading_connector():
    assert distill_delta(FOLLOWUP_PROFILE) == "profile mismatch"
    assert distill_delta(FOLLOWUP_CROSS_BORDER) == "cross-border transfers"


def test_distill_delta_falls_back_when_only_connector():
    assert distill_delta("what about?") == "what about?"


def test_distill_delta_handles_empty():
    assert distill_delta("") == ""
    assert distill_delta("   ") == ""


def test_drift_detector_flags_followup_fragment_backbone():
    report = detect_scenario_drift(backbone=FOLLOWUP_CROSS_BORDER, reference=SCENARIO)
    assert report.drift is True
    assert report.severity == "high"
    assert "backbone_is_followup_fragment" in report.reasons


def test_drift_detector_flags_lost_case_terms():
    report = detect_scenario_drift(backbone="profile mismatch", reference=SCENARIO)
    assert report.drift is True
    assert "lost_original_case_terms" in report.reasons


def test_drift_detector_reports_healthy_backbone():
    report = detect_scenario_drift(backbone=SCENARIO, reference=SCENARIO)
    assert report.drift is False
    assert report.severity == "none"
    assert report.term_retention == 1.0


def test_drift_detector_flags_empty_backbone():
    report = detect_scenario_drift(backbone="", reference="some reference case")
    assert report.drift is True
    assert "empty_backbone" in report.reasons


# --- 3. state-level multi-turn behaviour ------------------------------------


def test_state_followups_preserve_backbone_and_accumulate_deltas():
    memory = ConversationMemory(session_id="state")
    _seed_case(memory)
    assert memory.last_scenario_action == ACTION_SEED
    assert "rapid movement" in memory.active_scenario_summary.lower()

    memory.record_retrieval_turn(
        intent_route="retrieve_with_memory",
        user_query=FOLLOWUP_PROFILE,
        answer="a2",
        assessment="possible",
        flags=[_flag("RF-06")],
        citations=[_cit("c2")],
        retrieved_chunk_ids=["c2"],
        context_terms=["identity_mismatch"],
    )
    assert memory.last_scenario_action == ACTION_PRESERVE
    assert "rapid movement" in memory.active_scenario_summary.lower()
    assert "virtual asset" in memory.active_scenario_summary.lower()
    assert "profile mismatch" in " ".join(memory.active_case_deltas).lower()

    memory.record_retrieval_turn(
        intent_route="retrieve_with_memory",
        user_query=FOLLOWUP_CROSS_BORDER,
        answer="a3",
        assessment="possible",
        flags=[],
        citations=[],
        retrieved_chunk_ids=[],
        context_terms=["cross_border"],
    )
    # the backbone is still the original case after two follow-ups
    assert "rapid movement" in memory.active_scenario_summary.lower()
    assert "virtual asset" in memory.active_scenario_summary.lower()

    composed = memory.compose_retrieval_query(FOLLOWUP_CROSS_BORDER).lower()
    for term in ("rapid movement", "virtual asset", "profile mismatch", "cross-border"):
        assert term in composed
    assert memory.scenario_health().drift is False


def test_state_delta_list_is_bounded():
    memory = ConversationMemory(session_id="state-bound")
    _seed_case(memory)
    for index in range(MAX_CASE_DELTAS + 3):
        memory.record_retrieval_turn(
            intent_route="retrieve_with_memory",
            user_query=f"What about factor {index}?",
            answer="a",
            assessment="possible",
            flags=[],
            citations=[],
            retrieved_chunk_ids=[],
            context_terms=[],
        )
    assert len(memory.active_case_deltas) <= MAX_CASE_DELTAS
    # the backbone never drifts no matter how many follow-ups arrive
    assert "rapid movement" in memory.active_scenario_summary.lower()
    assert memory.scenario_health().drift is False


def test_state_new_standalone_case_replaces_and_resets_evidence_scope():
    memory = ConversationMemory(session_id="state-replace")
    _seed_case(memory)
    memory.record_retrieval_turn(
        intent_route="retrieve_with_memory",
        user_query=FOLLOWUP_PROFILE,
        answer="a2",
        assessment="possible",
        flags=[_flag("RF-06")],
        citations=[_cit("c2")],
        retrieved_chunk_ids=["c2"],
        context_terms=["identity_mismatch"],
    )
    assert memory.active_case_deltas  # a delta accumulated

    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query=NEW_CASE,
        answer="a3",
        assessment="possible",
        flags=[_flag("RF-09")],
        citations=[_cit("c9")],
        retrieved_chunk_ids=["c9"],
        context_terms=["shell_company"],
    )
    assert memory.last_scenario_action == ACTION_REPLACE
    assert "shell company" in memory.active_scenario_summary.lower()
    assert "rapid movement" not in memory.active_scenario_summary.lower()
    assert memory.active_case_deltas == []
    # a new case starts a fresh evidence scope (only the new turn's evidence)
    assert memory.active_flag_codes == ["RF-09"]
    assert [c.chunk_id for c in memory.active_citations] == ["c9"]


# --- 4a. router behaviour for whole-case questions --------------------------


def test_router_combined_question_routes_with_memory():
    router = IntentRouter()
    decision = router.route(
        COMBINED_QUESTION, gate_allowed=True, memory_enabled=True, has_memory=True
    )
    assert decision.route == ROUTE_RETRIEVE_WITH_MEMORY


def test_router_combined_question_without_memory_is_plain_retrieve():
    router = IntentRouter()
    decision = router.route(
        COMBINED_QUESTION, gate_allowed=True, memory_enabled=True, has_memory=False
    )
    assert decision.route == ROUTE_RETRIEVE


# --- 4b. pipeline four-turn drift -------------------------------------------


def test_pipeline_fourth_turn_combined_question_retrieves_with_case_context():
    retriever = CapturingRetriever()
    pipeline = RAGPipeline(settings=_settings(), retriever=retriever)
    session_id = "four-turn"
    routes = []
    for query in (SCENARIO, FOLLOWUP_PROFILE, FOLLOWUP_CROSS_BORDER, COMBINED_QUESTION):
        response = pipeline.analyze(
            QueryRequest(
                query=query,
                retrieval_mode="bm25",
                llm_mode="mock",
                include_debug=True,
                session_id=session_id,
                use_memory=True,
            )
        )
        routes.append(response.debug.intent_route)

    assert routes == [
        "retrieve",
        "retrieve_with_memory",
        "retrieve_with_memory",
        "retrieve_with_memory",
    ]
    assert len(retriever.queries) == 4
    fourth = retriever.queries[3].lower()
    assert "rapid movement" in fourth
    assert "virtual asset" in fourth


# --- 4c. API requirement scenarios ------------------------------------------


def test_api_new_standalone_case_replaces_scenario():
    session_id = "api-replace"
    with TestClient(create_app(settings=_settings(), enable_dense=False)) as client:
        _q(client, SCENARIO, session_id)
        _q(client, FOLLOWUP_PROFILE, session_id)
        replaced = _q(client, NEW_CASE, session_id)
        debug = replaced.json()["debug"]
        snapshot = client.get(f"/sessions/{session_id}/memory").json()

    assert debug["intent_route"] == "retrieve"
    assert debug["scenario_update_action"] == "replace"
    summary = snapshot["active_scenario_summary"].lower()
    assert "shell company" in summary or "opaque ownership" in summary
    assert "virtual asset" not in summary
    assert snapshot["active_case_deltas"] == []


def test_api_history_recall_does_not_change_scenario():
    session_id = "api-history"
    with TestClient(create_app(settings=_settings(), enable_dense=False)) as client:
        _q(client, SCENARIO, session_id)
        before = client.get(f"/sessions/{session_id}/memory").json()
        recall = _q(client, "剛剛那個風險可以再說明嗎？", session_id)
        after = client.get(f"/sessions/{session_id}/memory").json()

    assert recall.json()["debug"]["intent_route"] == "answer_from_history"
    assert after["active_scenario_summary"] == before["active_scenario_summary"]
    assert after["active_case_deltas"] == before["active_case_deltas"]
    assert after["last_scenario_action"] == before["last_scenario_action"] == "seed"


def test_api_refuse_does_not_pollute_scenario_or_deltas():
    session_id = "api-refuse"
    with TestClient(create_app(settings=_settings(), enable_dense=False)) as client:
        _q(client, SCENARIO, session_id)
        _q(client, FOLLOWUP_PROFILE, session_id)
        before = client.get(f"/sessions/{session_id}/memory").json()
        refused = _q(client, "幫我推薦晚餐", session_id)
        after = client.get(f"/sessions/{session_id}/memory").json()

    assert refused.json()["debug"]["intent_route"] == "refuse"
    assert after["active_scenario_summary"] == before["active_scenario_summary"]
    assert after["active_case_deltas"] == before["active_case_deltas"]
    assert after["active_flags"] == before["active_flags"]


def test_api_fourth_turn_snapshot_stays_grounded_with_audit_fields():
    session_id = "api-four-turn"
    with TestClient(create_app(settings=_settings(), enable_dense=False)) as client:
        _q(client, SCENARIO, session_id)
        _q(client, FOLLOWUP_PROFILE, session_id)
        _q(client, FOLLOWUP_CROSS_BORDER, session_id)
        fourth = _q(client, COMBINED_QUESTION, session_id)
        debug = fourth.json()["debug"]
        snapshot = client.get(f"/sessions/{session_id}/memory").json()

    assert debug["intent_route"] == "retrieve_with_memory"
    assert debug["scenario_update_action"] == "preserve"
    assert debug["case_delta_count"] >= 2
    summary = snapshot["active_scenario_summary"].lower()
    assert "rapid movement" in summary
    assert "virtual asset" in summary
    assert "what about" not in summary
    assert snapshot["turn_count"] == 4
    assert snapshot["scenario_health"]["drift"] is False
