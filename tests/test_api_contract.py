"""Contract tests for the FastAPI demo surface."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from rag_core.config import Settings


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "index"


def _client(artifact_dir: Path = ARTIFACT_DIR, **settings_overrides):
    settings = Settings(
        artifact_dir=str(artifact_dir),
        llm_mode="mock",
        enable_debug=True,
        **settings_overrides,
    )
    return TestClient(create_app(settings=settings, enable_dense=False))


def test_health_and_sources():
    with _client() as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["artifacts_loaded"] is True

        sources = client.get("/sources")
        assert sources.status_code == 200
        assert sources.json()["index_version"] == "demo-sample-v1"
        assert sources.json()["total_chunks"] == 12
        assert len(sources.json()["sources"]) == 3


def test_query_happy_path_and_fallback_labeling():
    with _client() as client:
        response = client.post(
            "/query",
            json={
                "query": "Funds show rapid movement through a virtual asset exchange.",
                "top_k": 5,
                "retrieval_mode": "hybrid",
                "llm_mode": "mock",
                "include_debug": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            "answer",
            "assessment",
            "identified_flags",
            "citations",
            "refusal",
            "debug",
        }
        assert body["assessment"] == "possible"
        assert body["citations"]
        assert body["refusal"]["refused"] is False
        assert body["debug"]["requested_mode"] == "hybrid"
        assert body["debug"]["retrieval_mode"] == "bm25"
        assert body["debug"]["fallback_used"] is True
        assert body["debug"]["bm25_used"] is True
        assert body["debug"]["rrf_used"] is False


def test_refusal_short_circuits_retrieval():
    with _client() as client:
        response = client.post(
            "/query",
            json={"query": "Assess this sanctions evasion case.", "include_debug": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["assessment"] == "refuse"
        assert body["refusal"]["refused"] is True
        assert body["citations"] == []
        assert body["debug"]["gate_decision"] == "refuse"
        assert body["debug"]["retrieval_mode"] == "none"
        assert body["debug"]["retrieved_chunk_ids"] == []


def test_debug_can_be_omitted():
    with _client() as client:
        response = client.post(
            "/query",
            json={"query": "rapid movement of funds", "include_debug": False},
        )
        assert response.status_code == 200
        assert response.json()["debug"] is None


def test_missing_artifacts_degrade_without_crashing(tmp_path):
    with _client(tmp_path / "missing") as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["artifacts_loaded"] is False

        query = client.post("/query", json={"query": "rapid movement"})
        assert query.status_code == 503
        assert query.json()["error"] == "ARTIFACTS_NOT_FOUND"
