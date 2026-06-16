"""Focused tests for deterministic and live-provider generation behavior."""

import json

from rag_core import generation
from rag_core.config import Settings
from rag_core.schemas import QueryRequest


QUERY = "Funds show rapid movement through an account."
CHUNKS = [
    {
        "chunk_id": "chunk-rf02",
        "source": "Demo source",
        "page": 1,
        "doc_category": "typology",
        "text": "Rapid movement of funds with little retention is a red flag.",
        "related_flags": ["RF-02"],
    }
]


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _google_payload(result):
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(result)}],
                }
            }
        ]
    }


def test_query_request_accepts_gemma_mode():
    request = QueryRequest(query=QUERY, llm_mode="gemma")

    assert request.llm_mode == "gemma"


def test_gemma_missing_key_falls_back_to_mock():
    result = generation.generate(
        query=QUERY,
        chunks=CHUNKS,
        llm_mode="gemma",
        model_name="some-gemma-model",
        gemini_api_key="",
    )

    debug = result["_generation_debug"]
    assert debug["effective_llm_mode"] == "mock"
    assert debug["fallback_used"] is True
    assert "API key is missing" in debug["fallback_reason"]


def test_gemma_missing_model_falls_back_to_mock():
    result = generation.generate(
        query=QUERY,
        chunks=CHUNKS,
        llm_mode="gemma",
        model_name="mock-local",
        gemini_api_key="fake",
    )

    debug = result["_generation_debug"]
    assert debug["effective_llm_mode"] == "mock"
    assert debug["fallback_used"] is True
    assert "MODEL_NAME must be set to an available Gemma model ID" in debug[
        "fallback_reason"
    ]


def test_gemma_success_uses_google_generate_content(monkeypatch):
    live_result = {
        "answer": "The evidence supports possible rapid movement.",
        "assessment": "possible",
        "identified_flags": [{"code": "RF-02"}],
        "citations": [{"chunk_id": "invented"}],
    }
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(_google_payload(live_result))

    monkeypatch.setattr(generation.httpx, "post", fake_post)

    result = generation.generate(
        query=QUERY,
        chunks=CHUNKS,
        llm_mode="gemma",
        model_name="some-gemma-model",
        gemini_api_key="fake",
    )

    assert calls[0][0].endswith("/models/some-gemma-model:generateContent")
    assert calls[0][1]["params"] == {"key": "fake"}
    assert result["assessment"] == "possible"
    assert result["citations"][0]["chunk_id"] == "chunk-rf02"
    assert result["_generation_debug"] == {
        "requested_llm_mode": "gemma",
        "effective_llm_mode": "gemma",
        "fallback_used": False,
        "fallback_reason": None,
    }


def test_generate_passes_configured_timeout_to_call_llm(monkeypatch):
    calls = []

    def fake_call_llm(system_prompt, user_prompt, llm_config, timeout=30.0):
        calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "llm_config": llm_config,
                "timeout": timeout,
            }
        )
        return {
            "answer": "The evidence supports possible rapid movement.",
            "assessment": "possible",
            "identified_flags": [{"code": "RF-02"}],
            "citations": [{"chunk_id": "invented"}],
        }

    monkeypatch.setattr(generation, "call_llm", fake_call_llm)

    result = generation.generate(
        query=QUERY,
        chunks=CHUNKS,
        llm_mode="gemma",
        model_name="some-gemma-model",
        gemini_api_key="fake",
        llm_timeout_seconds=123.0,
    )

    assert calls == [
        {
            "system_prompt": generation.SYSTEM_PROMPT,
            "user_prompt": generation.build_user_prompt(QUERY, CHUNKS),
            "llm_config": {
                "provider": "gemma",
                "llm_model_name": "some-gemma-model",
                "api_key": "fake",
            },
            "timeout": 123.0,
        }
    ]
    assert result["_generation_debug"]["effective_llm_mode"] == "gemma"


def test_gemma_malformed_google_response_falls_back_to_mock(monkeypatch):
    monkeypatch.setattr(
        generation.httpx,
        "post",
        lambda *args, **kwargs: _FakeResponse({"candidates": []}),
    )

    result = generation.generate(
        query=QUERY,
        chunks=CHUNKS,
        llm_mode="gemma",
        model_name="some-gemma-model",
        gemini_api_key="fake",
    )

    debug = result["_generation_debug"]
    assert debug["effective_llm_mode"] == "mock"
    assert debug["fallback_used"] is True
    assert debug["fallback_reason"]


def test_unsupported_provider_falls_back_to_mock():
    result = generation.generate(
        query=QUERY,
        chunks=CHUNKS,
        llm_mode="unsupported",
        model_name="some-model",
        groq_api_key="fake",
    )

    debug = result["_generation_debug"]
    assert debug["effective_llm_mode"] == "mock"
    assert debug["fallback_used"] is True
    assert debug["fallback_reason"] == "Unsupported LLM provider: unsupported"


def test_mock_mode_remains_deterministic_and_does_not_fallback(monkeypatch):
    monkeypatch.setattr(
        generation,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network call")),
    )

    first = generation.generate(query=QUERY, chunks=CHUNKS)
    second = generation.generate(query=QUERY, chunks=CHUNKS)

    assert first == second
    assert first["assessment"] == "possible"
    assert first["_generation_debug"]["fallback_used"] is False


def test_settings_exposes_llm_timeout_seconds(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
    settings = Settings()

    assert settings.llm_timeout_seconds == 120.0
