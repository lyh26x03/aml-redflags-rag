"""Focused tests for deterministic and live-provider generation behavior."""

import json

import httpx

from rag_core.error_sanitization import sanitize_error_message
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
PUBLIC_CHUNK_NO_FLAGS = {
    "chunk_id": "fatf_virtual_assets_red_flags.pdf_p15_c7",
    "source": "FATF",
    "page": 15,
    "doc_category": "sector_specific",
    "doc_type": "red_flag",
    "text": (
        "The account holder used mixing and immediately transferred funds out "
        "of the account through a virtual asset exchange."
    ),
}


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


def test_mock_generate_derives_chunk_flags_when_related_flags_are_absent():
    result = generation.mock_generate(
        query="Funds show rapid movement through an account and out to a virtual asset exchange.",
        chunks=[PUBLIC_CHUNK_NO_FLAGS],
    )

    assert result["assessment"] == "possible"
    assert {item["code"] for item in result["identified_flags"]} == {"RF-02", "RF-07"}
    assert [citation["chunk_id"] for citation in result["citations"]] == [
        "fatf_virtual_assets_red_flags.pdf_p15_c7"
    ]


def test_chunk_flags_keep_existing_related_flags_behavior():
    chunk = {
        **PUBLIC_CHUNK_NO_FLAGS,
        "related_flags": ["RF-04"],
    }

    assert generation._chunk_flags(chunk) == {"RF-04"}


def test_sanitize_error_message_redacts_query_api_key():
    sanitized = sanitize_error_message(
        "Provider failed at "
        "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key=SECRET123&alt=json"
    )

    assert sanitized is not None
    assert "SECRET123" not in sanitized
    assert "key=[redacted]" in sanitized


def test_sanitize_error_message_redacts_bearer_token():
    sanitized = sanitize_error_message(
        "Authorization: Bearer secret-token-value-123"
    )

    assert sanitized == "Authorization: Bearer [redacted]"


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
    assert debug["error_type"] == "missing_key"
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
    assert debug["error_type"] == "invalid_model_config"
    assert "MODEL_NAME must be set to an available Gemma model ID" in debug["fallback_reason"]


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
        "llm_model_name": "some-gemma-model",
        "fallback_used": False,
        "fallback_reason": None,
        "error_type": None,
        "http_status": None,
    }
    assert result["parse_success"] is True


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


def test_http_provider_fallback_reason_does_not_leak_raw_api_key(monkeypatch):
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=SECRET123",
    )
    response = httpx.Response(
        403,
        request=request,
        json={"error": {"message": "API key rejected"}},
    )

    def fake_call_llm(*args, **kwargs):
        raise httpx.HTTPStatusError(
            "Client error",
            request=request,
            response=response,
        )

    monkeypatch.setattr(generation, "call_llm", fake_call_llm)

    result = generation.generate(
        query=QUERY,
        chunks=CHUNKS,
        llm_mode="gemini",
        model_name="gemini-2.0-flash",
        gemini_api_key="fake",
    )

    debug = result["_generation_debug"]
    assert debug["fallback_used"] is True
    assert debug["error_type"] == "http_error"
    assert debug["http_status"] == 403
    assert "SECRET123" not in debug["fallback_reason"]
    assert "provider=gemini" in debug["fallback_reason"]
    assert "model=gemini-2.0-flash" in debug["fallback_reason"]


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
    assert debug["error_type"] == "parse_error"
    assert result["parse_success"] is False
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
    assert debug["error_type"] == "unsupported_provider"
    assert "Unsupported LLM provider: unsupported" in debug["fallback_reason"]


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
    assert first["parse_success"] is None
    assert first["_generation_debug"]["fallback_used"] is False


def test_settings_exposes_llm_timeout_seconds(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
    settings = Settings()

    assert settings.llm_timeout_seconds == 120.0
