"""Tests for the provider/mode matrix runner."""

import json
from datetime import datetime, timezone

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
        "provider_http_status": None,
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
    assert args.save_snapshot is False
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
    assert "- **Run ID:** `matrix-1`" in report
    assert "- **Modes requested:** mock" in report
    assert "- **Corpus profile:** public_226" in report
    assert "- **Corpus label:** public_226" in report
    assert "- **Total chunks:** 226" in report
    assert "## Summary By Mode" in report
    assert "## Per-Query Comparison" in report
    assert "Error type | Parse success" in report
    assert "provider/mode behavior smoke matrix" in report
    assert "Mock mode is deterministic" in report


def test_evaluate_query_mode_sanitizes_debug_fallback_reason_and_markdown(monkeypatch):
    response_body = {
        "assessment": "unlikely",
        "identified_flags": [],
        "citations": [],
        "parse_success": False,
        "debug": {
            "fallback_used": True,
            "fallback_reason": (
                "provider=gemini model=gemini-2.0-flash error_type=http_error "
                "message=https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.0-flash:generateContent?key=SECRET123"
            ),
            "error_type": "http_error",
            "http_status": 403,
            "llm_model_name": "gemini-2.0-flash",
            "retrieved_chunk_ids": ["chunk-1"],
        },
    }

    monkeypatch.setattr(
        run_model_matrix,
        "fetch_json",
        lambda *args, **kwargs: (200, response_body),
    )

    record = run_model_matrix.evaluate_query_mode(
        "matrix-1",
        "2026-06-16T00:00:00+00:00",
        "http://localhost:8000",
        "gemini",
        {"query_id": "case-1", "query": "Example query"},
        30.0,
        {"corpus_profile": "public_226", "total_chunks": 226},
    )
    report = run_model_matrix.render_markdown_report(
        [record],
        run_id="matrix-1",
        timestamp_utc="2026-06-16T00:00:00+00:00",
        base_url="http://localhost:8000",
        requested_modes=["gemini"],
        metadata={"corpus_profile": "public_226", "total_chunks": 226},
    )

    assert record["fallback_reason"] is not None
    assert "SECRET123" not in record["fallback_reason"]
    assert record["error_type"] == "http_error"
    assert record["parse_success"] is False
    assert record["provider_http_status"] == 403
    assert record["model"] == "gemini-2.0-flash"
    assert "SECRET123" not in report


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


def test_resolve_snapshot_paths_uses_archive_dir_and_increments_revision(tmp_path):
    out_jsonl = tmp_path / "results" / "model_matrix_latest.jsonl"
    out_md = tmp_path / "reports" / "model_matrix_latest.md"
    timestamp = datetime(2026, 6, 16, tzinfo=timezone.utc)

    first_jsonl, first_md = run_model_matrix.resolve_snapshot_paths(
        out_jsonl,
        out_md,
        timestamp=timestamp,
        requested_modes=["mock", "gemini"],
        corpus_label="public_226",
        corpus_profile="sample",
    )
    assert first_jsonl == (
        tmp_path
        / "results"
        / "archive"
        / "model_matrix_20260616_public_226_mock-gemini.jsonl"
    )
    assert first_md == (
        tmp_path
        / "reports"
        / "archive"
        / "model_matrix_20260616_public_226_mock-gemini.md"
    )

    first_jsonl.parent.mkdir(parents=True, exist_ok=True)
    first_jsonl.write_text("existing", encoding="utf-8")

    second_jsonl, second_md = run_model_matrix.resolve_snapshot_paths(
        out_jsonl,
        out_md,
        timestamp=timestamp,
        requested_modes=["mock", "gemini"],
        corpus_label="public_226",
        corpus_profile="sample",
    )
    assert second_jsonl.name == "model_matrix_20260616_public_226_mock-gemini_r2.jsonl"
    assert second_md.name == "model_matrix_20260616_public_226_mock-gemini_r2.md"


def test_main_writes_latest_outputs_without_snapshot(tmp_path, monkeypatch):
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "case-1",
                    "query": "Example query",
                }
            ]
        ),
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "results" / "model_matrix_latest.jsonl"
    out_md = tmp_path / "reports" / "model_matrix_latest.md"

    monkeypatch.setattr(
        run_model_matrix,
        "fetch_service_metadata",
        lambda *args, **kwargs: {"corpus_profile": "public_226", "total_chunks": 226},
    )
    monkeypatch.setattr(
        run_model_matrix,
        "evaluate_query_mode",
        lambda run_id, timestamp_utc, base_url, mode, query_case, timeout, metadata: _record(
            mode,
            "ok",
            12,
            query_id=query_case["query_id"],
            query=query_case["query"],
            corpus_profile=metadata["corpus_profile"],
            total_chunks=metadata["total_chunks"],
        ),
    )

    exit_code = run_model_matrix.main(
        [
            "--queries",
            str(queries_path),
            "--out-jsonl",
            str(out_jsonl),
            "--out-md",
            str(out_md),
            "--modes",
            "mock,gemini",
        ]
    )

    assert exit_code == 0
    assert out_jsonl.exists()
    assert out_md.exists()
    assert not (tmp_path / "results" / "archive").exists()
    assert not (tmp_path / "reports" / "archive").exists()


def test_main_with_save_snapshot_writes_archive_copy(tmp_path, monkeypatch):
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps([{"query_id": "case-1", "query": "Example query"}]),
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "results" / "model_matrix_latest.jsonl"
    out_md = tmp_path / "reports" / "model_matrix_latest.md"

    monkeypatch.setattr(
        run_model_matrix,
        "fetch_service_metadata",
        lambda *args, **kwargs: {"corpus_profile": "public_226", "total_chunks": 226},
    )
    monkeypatch.setattr(
        run_model_matrix,
        "evaluate_query_mode",
        lambda run_id, timestamp_utc, base_url, mode, query_case, timeout, metadata: _record(
            mode,
            "ok",
            12,
            query_id=query_case["query_id"],
            query=query_case["query"],
            corpus_profile=metadata["corpus_profile"],
            total_chunks=metadata["total_chunks"],
        ),
    )
    fixed_timestamp = datetime(2026, 6, 16, tzinfo=timezone.utc)

    class _FixedDateTime:
        @staticmethod
        def now(tz=None):
            if tz is None:
                return fixed_timestamp.replace(tzinfo=None)
            return fixed_timestamp.astimezone(tz)

    monkeypatch.setattr(run_model_matrix, "datetime", _FixedDateTime)

    exit_code = run_model_matrix.main(
        [
            "--queries",
            str(queries_path),
            "--out-jsonl",
            str(out_jsonl),
            "--out-md",
            str(out_md),
            "--modes",
            "mock",
            "--corpus-label",
            "public_226",
            "--save-snapshot",
        ]
    )

    snapshot_jsonl = (
        tmp_path
        / "results"
        / "archive"
        / "model_matrix_20260616_public_226_mock.jsonl"
    )
    snapshot_md = (
        tmp_path
        / "reports"
        / "archive"
        / "model_matrix_20260616_public_226_mock.md"
    )

    assert exit_code == 0
    assert out_jsonl.exists()
    assert out_md.exists()
    assert snapshot_jsonl.exists()
    assert snapshot_md.exists()
