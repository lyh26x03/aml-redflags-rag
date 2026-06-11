"""HTTP smoke test for a running AML red-flag RAG API."""

import os

import httpx


BASE_URL = os.getenv("SMOKE_BASE_URL", "http://localhost:8000").rstrip("/")


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["status"] == "ok"
        assert health.json()["artifacts_loaded"] is True

        query = client.post(
            "/query",
            json={
                "query": "Funds show rapid movement through a virtual asset exchange.",
                "top_k": 5,
                "retrieval_mode": "hybrid",
                "llm_mode": "mock",
                "include_debug": True,
            },
        )
        query.raise_for_status()
        body = query.json()
        assert body["answer"]
        assert body["assessment"] == "possible"
        assert body["citations"]
        assert body["debug"]["requested_mode"] == "hybrid"

        refusal = client.post(
            "/query",
            json={"query": "Assess this sanctions evasion case.", "include_debug": True},
        )
        refusal.raise_for_status()
        refused = refusal.json()
        assert refused["assessment"] == "refuse"
        assert refused["refusal"]["refused"] is True
        assert refused["citations"] == []

        sources = client.get("/sources")
        sources.raise_for_status()
        assert sources.json()["total_chunks"] > 0
        assert sources.json()["sources"]

    print(f"Smoke test passed against {BASE_URL}")


if __name__ == "__main__":
    main()
