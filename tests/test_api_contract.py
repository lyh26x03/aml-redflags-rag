"""Contract tests for the FastAPI demo surface."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from rag_core.config import Settings


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "index"
PUBLIC_226_DIR = REPO_ROOT / "data" / "public_corpus_226"


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
        assert health.json()["corpus_profile"] == "sample"
        assert health.json()["artifacts_loaded"] is True
        assert health.json()["chunk_count"] == 12
        assert health.json()["source_count"] == 3

        sources = client.get("/sources")
        assert sources.status_code == 200
        assert sources.json()["corpus_profile"] == "sample"
        assert sources.json()["index_version"] == "demo-sample-v1"
        assert sources.json()["chunk_count"] == 12
        assert sources.json()["total_chunks"] == 12
        assert sources.json()["source_count"] == 3
        assert len(sources.json()["sources"]) == 3


def test_public_226_profile_loads_and_reports_sources():
    with _client(
        artifact_dir=ARTIFACT_DIR,
        corpus_profile="public_226",
        public_226_artifact_dir=str(PUBLIC_226_DIR),
    ) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["corpus_profile"] == "public_226"
        assert health.json()["index_version"] == "public-226-v2"
        assert health.json()["chunk_count"] == 226
        assert health.json()["source_count"] == 3
        assert health.json()["source_names"] == [
            "FATF Trade-Based Money Laundering Risk Indicators",
            "FATF Virtual Assets Red Flag Indicators",
            "Taiwan AML Training Slides",
        ]

        sources = client.get("/sources")
        assert sources.status_code == 200
        body = sources.json()
        assert body["corpus_profile"] == "public_226"
        assert body["index_version"] == "public-226-v2"
        assert body["chunk_count"] == 226
        assert body["total_chunks"] == 226
        assert body["source_count"] == 3
        assert [source["chunk_count"] for source in body["sources"]] == [51, 165, 10]


def test_query_works_in_mock_mode_with_public_226_profile():
    with _client(
        artifact_dir=ARTIFACT_DIR,
        corpus_profile="public_226",
        public_226_artifact_dir=str(PUBLIC_226_DIR),
    ) as client:
        response = client.post(
            "/query",
            json={
                "query": "A virtual asset exchange customer uses a mixer and rapidly transfers funds.",
                "top_k": 3,
                "retrieval_mode": "bm25",
                "llm_mode": "mock",
                "include_debug": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["assessment"] in {"possible", "unlikely"}
        assert body["refusal"]["refused"] is False
        assert body["debug"]["retrieval_mode"] == "bm25"
        assert body["debug"]["bm25_used"] is True
        assert len(body["debug"]["retrieved_chunk_ids"]) == 3


def test_query_happy_path_and_fallback_labeling():
    # The query contains "rapid movement" (→ RF-02) and "virtual asset exchange"
    # (→ RF-07).  The sample corpus in artifacts/index/chunks.json includes chunks
    # tagged with both RF-02 and RF-07, so mock_generate produces "possible".
    # If the sample corpus is rebuilt without those flags, this assertion will need
    # updating.
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


def test_include_debug_defaults_to_true_when_setting_enabled():
    # include_debug omitted from request; with enable_debug=True in settings,
    # pipeline includes debug because `None is not False` evaluates True.
    with _client() as client:
        response = client.post(
            "/query",
            json={"query": "rapid movement of funds"},
        )
        assert response.status_code == 200
        assert response.json()["debug"] is not None


def test_missing_artifacts_degrade_without_crashing(tmp_path):
    with _client(tmp_path / "missing") as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["artifacts_loaded"] is False

        query = client.post("/query", json={"query": "rapid movement"})
        assert query.status_code == 503
        assert query.json()["error"] == "ARTIFACTS_NOT_FOUND"
