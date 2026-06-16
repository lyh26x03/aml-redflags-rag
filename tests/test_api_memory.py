"""Integration tests for intent routing + structured conversation memory."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from rag_core.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "index"

SCENARIO = "Funds show rapid movement through a virtual asset exchange."


def _client(**settings_overrides):
    settings = Settings(
        artifact_dir=str(ARTIFACT_DIR),
        llm_mode="mock",
        model_name="mock-local",
        enable_debug=True,
        **settings_overrides,
    )
    return TestClient(create_app(settings=settings, enable_dense=False))


def _query(client, query, **extra):
    payload = {
        "query": query,
        "retrieval_mode": "bm25",
        "llm_mode": "mock",
        "include_debug": True,
    }
    payload.update(extra)
    return client.post("/query", json=payload)


# --- backward compatibility --------------------------------------------------


def test_single_turn_query_remains_backward_compatible():
    with _client() as client:
        response = _query(client, SCENARIO)
        assert response.status_code == 200
        body = response.json()
        # exact top-level key set is unchanged
        assert set(body) == {
            "answer",
            "assessment",
            "identified_flags",
            "citations",
            "refusal",
            "parse_success",
            "debug",
        }
        debug = body["debug"]
        assert debug["intent_route"] == "retrieve"
        assert debug["memory_used"] is False
        assert debug["memory_updated"] is False


def test_use_memory_false_does_not_create_or_update_memory():
    with _client() as client:
        response = _query(client, SCENARIO, session_id="s1", use_memory=False)
        assert response.status_code == 200
        assert response.json()["debug"]["memory_updated"] is False
        # no session memory was created
        assert client.get("/sessions/s1/memory").status_code == 404


def test_memory_mode_off_disables_memory_even_when_use_memory_true():
    with _client() as client:
        response = _query(
            client, SCENARIO, session_id="s1", use_memory=True, memory_mode="off"
        )
        assert response.status_code == 200
        assert response.json()["debug"]["memory_used"] is False
        assert client.get("/sessions/s1/memory").status_code == 404


# --- memory updates ----------------------------------------------------------


def test_use_memory_true_with_session_updates_memory():
    with _client() as client:
        response = _query(client, SCENARIO, session_id="s1", use_memory=True)
        assert response.status_code == 200
        debug = response.json()["debug"]
        assert debug["intent_route"] == "retrieve"
        assert debug["memory_updated"] is True
        assert debug["memory_turn_count"] == 1
        assert debug["active_flags"]  # at least one flag retained

        snapshot = client.get("/sessions/s1/memory").json()
        assert snapshot["turn_count"] == 1
        assert snapshot["active_scenario_summary"]
        assert snapshot["active_flags"]


def test_followup_recalls_previous_flags_from_memory():
    with _client() as client:
        _query(client, SCENARIO, session_id="s1", use_memory=True)
        response = _query(
            client, "剛剛那個風險可以再說明嗎？", session_id="s1", use_memory=True
        )
        assert response.status_code == 200
        body = response.json()
        debug = body["debug"]
        assert debug["intent_route"] == "answer_from_history"
        assert debug["memory_used"] is True
        assert debug["referenced_previous_answer"] is True
        assert debug["active_flags"]
        # the recalled flags are surfaced as identified_flags
        assert body["identified_flags"]


def test_followup_recalls_previous_citations_from_memory():
    with _client() as client:
        _query(client, SCENARIO, session_id="s1", use_memory=True)
        response = _query(
            client, "剛剛引用的是哪些來源？", session_id="s1", use_memory=True
        )
        assert response.status_code == 200
        debug = response.json()["debug"]
        assert debug["intent_route"] == "answer_from_history"
        assert debug["referenced_previous_evidence"] is True
        assert debug["active_citation_count"] >= 1
        assert response.json()["citations"]


def test_followup_question_uses_retrieve_with_memory():
    with _client() as client:
        _query(client, SCENARIO, session_id="s1", use_memory=True)
        response = _query(
            client, "那跟客戶職業不符有關嗎？", session_id="s1", use_memory=True
        )
        debug = response.json()["debug"]
        assert debug["intent_route"] == "retrieve_with_memory"
        assert debug["memory_used"] is True
        # composing prior scenario + new question surfaces the mismatch flag
        codes = {flag["code"] for flag in response.json()["identified_flags"]}
        assert "RF-06" in codes


# --- answer_from_history with no memory --------------------------------------


def test_answer_from_history_without_memory_gives_no_context_response():
    with _client() as client:
        response = _query(
            client, "剛剛引用的是哪些來源？", session_id="fresh", use_memory=True
        )
        assert response.status_code == 200
        body = response.json()
        debug = body["debug"]
        assert debug["intent_route"] == "answer_from_history"
        assert debug["memory_available"] is False
        assert debug["referenced_previous_answer"] is False
        assert body["refusal"]["refused"] is False
        assert "先前" in body["answer"] or "No previous" in body["answer"]


# --- clarify -----------------------------------------------------------------


def test_vague_first_turn_routes_to_clarify_and_stores_unresolved():
    with _client() as client:
        response = _query(client, "這樣有沒有問題？", session_id="s1", use_memory=True)
        body = response.json()
        debug = body["debug"]
        assert debug["intent_route"] == "clarify"
        assert body["assessment"] == "unlikely"
        assert body["refusal"]["refused"] is False
        assert body["identified_flags"] == []

        snapshot = client.get("/sessions/s1/memory").json()
        assert snapshot["unresolved_questions"]


# --- refuse: no memory pollution ---------------------------------------------


def test_out_of_scope_refuses_and_does_not_pollute_memory():
    with _client() as client:
        # establish a real scenario first
        _query(client, SCENARIO, session_id="s1", use_memory=True)
        before = client.get("/sessions/s1/memory").json()

        response = _query(client, "幫我推薦晚餐", session_id="s1", use_memory=True)
        body = response.json()
        debug = body["debug"]
        assert debug["intent_route"] == "refuse"
        assert body["assessment"] == "refuse"
        assert body["refusal"]["refused"] is True
        assert debug["memory_updated"] is False

        after = client.get("/sessions/s1/memory").json()
        # the refusal left the active scenario state untouched
        assert after["turn_count"] == before["turn_count"]
        assert after["active_flags"] == before["active_flags"]
        assert after["active_scenario_summary"] == before["active_scenario_summary"]


def test_refuse_on_empty_session_does_not_create_active_scenario():
    with _client() as client:
        response = _query(client, "幫我推薦晚餐", session_id="s1", use_memory=True)
        assert response.json()["debug"]["intent_route"] == "refuse"
        # no memory was created by the refusal
        assert client.get("/sessions/s1/memory").status_code == 404


# --- gate-based refusal still works in memory mode ---------------------------


def test_gate_refusal_in_memory_mode_routes_to_refuse():
    with _client() as client:
        response = _query(
            client, "Assess this sanctions evasion case.", session_id="s1", use_memory=True
        )
        body = response.json()
        assert body["assessment"] == "refuse"
        assert body["debug"]["intent_route"] == "refuse"
        assert body["debug"]["gate_decision"] == "refuse"


# --- bounds across many turns ------------------------------------------------


def test_memory_stays_bounded_across_many_turns():
    with _client() as client:
        for _ in range(12):
            _query(client, SCENARIO, session_id="s1", use_memory=True)
        snapshot = client.get("/sessions/s1/memory").json()
        assert snapshot["turn_count"] == 12
        assert len(snapshot["recent_turns"]) <= 8
        assert len(snapshot["active_citations"]) <= 10


# --- endpoints ---------------------------------------------------------------


def test_memory_inspection_endpoint_returns_snapshot():
    with _client() as client:
        _query(client, SCENARIO, session_id="s1", use_memory=True)
        response = client.get("/sessions/s1/memory")
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "s1"
        assert body["turn_count"] == 1
        assert "recent_turns" in body


def test_memory_inspection_endpoint_404_for_unknown_session():
    with _client() as client:
        response = client.get("/sessions/nope/memory")
        assert response.status_code == 404
        assert response.json()["error"] == "SESSION_NOT_FOUND"


def test_memory_delete_endpoint_clears_session():
    with _client() as client:
        _query(client, SCENARIO, session_id="s1", use_memory=True)
        deleted = client.delete("/sessions/s1/memory")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert client.get("/sessions/s1/memory").status_code == 404
        # deleting again reports nothing to delete
        assert client.delete("/sessions/s1/memory").json()["deleted"] is False


def test_reset_memory_flag_clears_before_processing():
    with _client() as client:
        _query(client, SCENARIO, session_id="s1", use_memory=True)
        _query(client, SCENARIO, session_id="s1", use_memory=True)
        # reset_memory on a new turn should restart the count at 1
        response = _query(
            client, SCENARIO, session_id="s1", use_memory=True, reset_memory=True
        )
        assert response.json()["debug"]["memory_turn_count"] == 1
