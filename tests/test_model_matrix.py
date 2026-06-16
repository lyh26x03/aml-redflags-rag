"""Tests for the provider/mode matrix runner."""

import json

from scripts import run_model_matrix


def _record(mode, status, latency_ms, fallback_used=False, **overrides):
    record = {
        "run_id": "matrix-1",
        "timestamp_utc": "2026-06-16T00:00:00+00:00",
        "query_id": "case-1",
        "query": "Example query",
        "llm_mode": mode,
        "status": status,
        "http_status": 200 if status == "ok" else 422,
        "latency_ms": latency_ms,
        "assessment": "possible" if status == "ok" else None,
        "identified_flag_codes": ["RF-02"] if status == "ok" else [],
        "citation_count": 1 if status == "ok" else 0,
        "retrieved_chunk_ids": ["chunk-1"] if status == "ok" else [],
        "fallback_used": fallback_used,
        "fallback_reason": None,
        "parse_success": None,
        "model": None,
        "provider": mode,
        "corpus_profile": "public_226",
        "total_chunks": 226,
        "error_type": None,
        "error_message": None,
    }
    record.update(overrides)
    return record


def test_parse_args_defaults_to_mock_only():
    args = run_model_matrix.parse_args([])

    assert args.modes == "mock"
    assert run_model_matrix.parse_modes(args.modes) == ["mock"]


def test_summarize_results_counts_statuses_and_latency():
    records = [
        _record("mock", "ok", 10),
        _record("mock", "ok", 30, fallback_used=True, query_id="case-2"),
        _record("gemini", "unsupported", 20),
        _record("gemini", "service_error", 40, http_status=503),
    ]

    summary = {
        item["llm_mode"]: item
        for item in run_model_matrix.summarize_results(records)
    }

    assert summary["mock"] == {
        "llm_mode": "mock",
        "total": 2,
        "ok": 2,
        "unsupported": 0,
        "errors": 0,
        "fallback_count": 1,
        "avg_latency_ms": 20.0,
    }
    assert summary["gemini"] == {
        "llm_mode": "gemini",
        "total": 2,
        "ok": 0,
        "unsupported": 1,
        "errors": 1,
        "fallback_count": 0,
        "avg_latency_ms": 30.0,
    }


def test_markdown_renderer_includes_required_sections_and_note():
    report = run_model_matrix.render_markdown_report(
        [_record("mock", "ok", 12, fallback_used=False)],
        run_id="matrix-1",
        timestamp_utc="2026-06-16T00:00:00+00:00",
        base_url="http://localhost:8000",
        requested_modes=["mock"],
        metadata={"corpus_profile": "public_226", "total_chunks": 226},
        corpus_label="public_226",
    )

    assert report.startswith("# Model Matrix Runner Report\n")
    assert "## Summary By Mode" in report
    assert "## Per-Query Comparison" in report
    assert "provider/mode behavior smoke matrix" in report
    assert "Mock mode is deterministic" in report


def test_unsupported_mode_helper_detects_llm_mode_validation_error():
    body = {
        "detail": [
            {
                "type": "literal_error",
                "loc": ["body", "llm_mode"],
                "msg": "Input should be 'mock', 'gemini', 'gemma' or 'groq'",
            }
        ]
    }

    assert run_model_matrix.is_unsupported_llm_mode(422, body) is True
    assert run_model_matrix.is_unsupported_llm_mode(
        422,
        {"detail": [{"loc": ["body", "query"], "msg": "Field required"}]},
    ) is False


def test_jsonl_writer_writes_line_oriented_json(tmp_path):
    records = [
        _record("mock", "ok", 10),
        _record("gemini", "unsupported", 20, query_id="case-2"),
    ]
    output = tmp_path / "nested" / "model-matrix.jsonl"

    run_model_matrix.write_jsonl(output, records)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records
