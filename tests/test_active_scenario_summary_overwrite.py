"""Regression tests for active_scenario_summary drift in memory follow-ups.

These tests encode the expected role of active_scenario_summary as the active
AML case summary, not the latest short follow-up question.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from rag_core.config import Settings
from rag_core.memory import ConversationMemory
from rag_core.pipeline import RAGPipeline
from rag_core.retrieval import RetrievalResult
from rag_core.schemas import QueryRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "index"

SCENARIO = "Funds show rapid movement through a virtual asset exchange."
FOLLOWUP_PROFILE = "What about profile mismatch?"
FOLLOWUP_CROSS_BORDER = "What about cross-border transfers?"


def _flag(code, name):
    return {"code": code, "name": name, "name_zh": name, "reason": "diagnostic"}


def _citation(chunk_id, excerpt):
    return {"chunk_id": chunk_id, "source": "diagnostic", "excerpt": excerpt}


class CapturingRetriever:
    """Minimal retriever that records the query string passed by the pipeline."""

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
                        "cross-border transfers"
                    ),
                    "score": 1.0,
                    "doc_category": "aml",
                    "doc_type": "diagnostic",
                    "explanation_style": "neutral",
                    "related_flags": ["RF-02", "RF-05", "RF-06", "RF-07"],
                }
            ],
            requested_mode=requested_mode,
            effective_mode="bm25",
            bm25_used=True,
        )


def _settings():
    return Settings(
        artifact_dir=str(ARTIFACT_DIR),
        llm_mode="mock",
        model_name="mock-local",
        enable_debug=True,
    )


def _query(client, query, session_id="summary-overwrite-api"):
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


def test_retrieve_with_memory_does_not_replace_case_summary_with_followup():
    memory = ConversationMemory(session_id="summary-overwrite-state")
    memory.record_retrieval_turn(
        intent_route="retrieve",
        user_query=SCENARIO,
        answer="turn 1 answer",
        assessment="possible",
        flags=[
            _flag("RF-02", "Rapid Movement"),
            _flag("RF-07", "Virtual Asset Anonymity"),
        ],
        citations=[_citation("c-1", "rapid movement through virtual assets")],
        retrieved_chunk_ids=["c-1"],
        context_terms=["rapid_movement", "virtual_assets"],
    )

    memory.record_retrieval_turn(
        intent_route="retrieve_with_memory",
        user_query=FOLLOWUP_PROFILE,
        answer="turn 2 answer",
        assessment="possible",
        flags=[_flag("RF-06", "Profile Mismatch")],
        citations=[_citation("c-2", "profile mismatch evidence")],
        retrieved_chunk_ids=["c-2"],
        context_terms=["identity_mismatch"],
    )

    assert "rapid movement" in memory.active_scenario_summary.lower()
    assert "virtual asset" in memory.active_scenario_summary.lower()
    assert memory.active_scenario_summary != FOLLOWUP_PROFILE


def test_pipeline_third_followup_retrieval_query_retains_case_summary_terms():
    retriever = CapturingRetriever()
    pipeline = RAGPipeline(settings=_settings(), retriever=retriever)
    session_id = "summary-overwrite-pipeline"

    for query in (SCENARIO, FOLLOWUP_PROFILE, FOLLOWUP_CROSS_BORDER):
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
        assert response.debug is not None

    assert len(retriever.queries) == 3
    assert "rapid movement" in retriever.queries[1].lower()
    assert "virtual asset" in retriever.queries[1].lower()
    assert "profile mismatch" in retriever.queries[1].lower()

    third_query = retriever.queries[2].lower()
    assert "rapid movement" in third_query
    assert "virtual asset" in third_query
    assert "cross-border" in third_query


def test_api_memory_snapshot_keeps_case_summary_after_consecutive_followups():
    with TestClient(create_app(settings=_settings(), enable_dense=False)) as client:
        first = _query(client, SCENARIO)
        second = _query(client, FOLLOWUP_PROFILE)
        third = _query(client, FOLLOWUP_CROSS_BORDER)

        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 200
        assert second.json()["debug"]["intent_route"] == "retrieve_with_memory"
        assert third.json()["debug"]["intent_route"] == "retrieve_with_memory"

        snapshot = client.get("/sessions/summary-overwrite-api/memory").json()

    summary = snapshot["active_scenario_summary"].lower()
    assert snapshot["turn_count"] == 3
    assert "rapid movement" in summary
    assert "virtual asset" in summary
    assert "what about" not in summary
