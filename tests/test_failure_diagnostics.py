"""Tests for Failure Diagnostics Lite."""

import json
from datetime import datetime, timezone

from scripts import run_failure_diagnostics


def _summary(**overrides):
    summary = {
        "record_type": "group_summary",
        "group_id": "synthetic-group",
        "expected_behavior": "stable_possible",
        "assessment_consistency": 1.0,
        "refusal_consistency": 1.0,
        "flag_jaccard_avg": 1.0,
        "citation_jaccard_avg": 1.0,
        "retrieved_chunk_jaccard_avg": 1.0,
        "set_consistency_threshold": 0.5,
        "passed": True,
    }
    summary.update(overrides)
    return summary


def _categories(issues):
    return {item["category"] for item in issues}


def test_missing_inputs_produce_report_and_do_not_crash(tmp_path):
    issues, inputs, api_records, cqc_records = run_failure_diagnostics.collect_diagnostics(
        tmp_path / "missing-api.jsonl", tmp_path / "missing-cqc.jsonl"
    )
    report = run_failure_diagnostics.render_markdown_report(
        issues, inputs, api_records, cqc_records
    )

    assert _categories(issues) == {"SERVICE_UNREACHABLE_OR_MISSING_INPUT"}
    assert all(item["severity"] == "MEDIUM" for item in issues)
    assert "# Failure Diagnostics Lite Report" in report
    assert "unavailable" in report


def test_consistent_cqc_summary_produces_no_high_issues():
    issues = run_failure_diagnostics.diagnose_cqc([_summary()])

    assert not any(item["severity"] == "HIGH" for item in issues)


def test_low_citation_overlap_is_diagnosed():
    issues = run_failure_diagnostics.diagnose_cqc(
        [_summary(citation_jaccard_avg=0.25)]
    )

    assert "LOW_CITATION_OVERLAP" in _categories(issues)


def test_low_retrieved_chunk_overlap_is_diagnosed():
    issues = run_failure_diagnostics.diagnose_cqc(
        [_summary(retrieved_chunk_jaccard_avg=0.25)]
    )

    assert "RETRIEVED_CHUNK_INSTABILITY" in _categories(issues)


def test_low_flag_overlap_is_diagnosed():
    issues = run_failure_diagnostics.diagnose_cqc([_summary(flag_jaccard_avg=0.25)])

    assert "IDENTIFIED_FLAG_INSTABILITY" in _categories(issues)


def test_assessment_instability_is_high_severity():
    issues = run_failure_diagnostics.diagnose_cqc(
        [_summary(assessment_consistency=0.75)]
    )

    assert any(
        item["category"] == "ASSESSMENT_INSTABILITY"
        and item["severity"] == "HIGH"
        for item in issues
    )


def test_refusal_instability_is_high_severity():
    issues = run_failure_diagnostics.diagnose_cqc(
        [_summary(refusal_consistency=0.75)]
    )

    assert any(
        item["category"] == "REFUSAL_INSTABILITY" and item["severity"] == "HIGH"
        for item in issues
    )


def test_api_smoke_errors_produce_failure():
    issues = run_failure_diagnostics.diagnose_api_smoke(
        [{"id": "case-1", "passed": False, "status_code": 500, "errors": ["failed"]}]
    )

    assert "API_SMOKE_FAILURE" in _categories(issues)


def test_markdown_renderer_includes_required_sections_and_category():
    issues = run_failure_diagnostics.diagnose_cqc(
        [_summary(citation_jaccard_avg=0.25)]
    )
    report = run_failure_diagnostics.render_markdown_report(
        issues,
        [{"source": "api_smoke", "path": "api.jsonl", "found": True, "record_count": 1}],
        [],
        [_summary(citation_jaccard_avg=0.25)],
        datetime(2026, 6, 15, 8, 0, tzinfo=timezone.utc),
    )

    assert report.startswith("# Failure Diagnostics Lite Report\n")
    assert "## Issue Counts By Severity" in report
    assert "## Issue Counts By Category" in report
    assert "LOW_CITATION_OVERLAP" in report
    assert "## Interpretation Note" in report
    assert "not a model-quality benchmark" in report


def test_jsonl_writer_writes_line_oriented_json(tmp_path):
    issues = [
        run_failure_diagnostics.issue("cqc", "MEDIUM", "LOW_CITATION_OVERLAP", "one"),
        run_failure_diagnostics.issue("api_smoke", "HIGH", "API_SMOKE_FAILURE", "two"),
    ]
    output = tmp_path / "nested" / "diagnostics.jsonl"

    run_failure_diagnostics.write_jsonl(output, issues)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == issues


def test_no_issues_writes_empty_jsonl(tmp_path):
    output = tmp_path / "diagnostics.jsonl"

    run_failure_diagnostics.write_jsonl(output, [])

    assert output.read_text(encoding="utf-8") == ""
