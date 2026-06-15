"""Tests for the CQC-RAG Lite evaluation harness."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import run_cqc_eval


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = REPO_ROOT / "eval" / "queries" / "cqc_scenarios_5.json"


def _valid_group() -> dict:
    return {
        "group_id": "synthetic-group",
        "description": "A synthetic consistency scenario.",
        "expected_behavior": "stable_possible",
        "variants": [
            {"variant_id": f"synthetic-{index}", "query": f"Query variant {index}"}
            for index in range(1, 5)
        ],
    }


def _record(**overrides) -> dict:
    record = {
        "assessment": "possible",
        "refusal_refused": False,
        "identified_flag_codes": ["RF-01", "RF-02"],
        "citation_chunk_ids": ["chunk-1", "chunk-2"],
        "retrieved_chunk_ids": ["chunk-1", "chunk-2"],
        "errors": [],
    }
    record.update(overrides)
    return record


def test_loads_committed_scenario_groups():
    groups = run_cqc_eval.load_groups(SCENARIOS_PATH)

    assert len(groups) == 5
    group_ids = [group["group_id"] for group in groups]
    assert all(group_id.strip() for group_id in group_ids)
    assert len(group_ids) == len(set(group_ids))
    assert all(len(group["variants"]) == 4 for group in groups)
    assert all(
        isinstance(variant["query"], str) and variant["query"].strip()
        for group in groups
        for variant in group["variants"]
    )


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ({}, "input must be a JSON array"),
        ([{"group_id": ""}], "non-empty group_id"),
        ([_valid_group() | {"variants": []}], "variants must contain 3 to 5 items"),
    ],
)
def test_malformed_scenario_data_fails_clearly(groups, message):
    with pytest.raises(ValueError, match=message):
        run_cqc_eval.validate_groups(groups)


def test_duplicate_group_ids_fail_clearly():
    duplicate = copy.deepcopy(_valid_group())

    with pytest.raises(ValueError, match="duplicate group_id: synthetic-group"):
        run_cqc_eval.validate_groups([_valid_group(), duplicate])


def test_duplicate_variant_ids_fail_clearly():
    group = _valid_group()
    group["variants"][1]["variant_id"] = group["variants"][0]["variant_id"]

    with pytest.raises(ValueError, match="duplicate variant_id: synthetic-1"):
        run_cqc_eval.validate_groups([group])


def test_summarize_group_passes_consistent_synthetic_responses():
    summary = run_cqc_eval.summarize_group(_valid_group(), [_record() for _ in range(4)])

    assert summary["variant_count"] == 4
    assert summary["assessment_consistency"] == 1.0
    assert summary["refusal_consistency"] == 1.0
    assert summary["flag_jaccard_avg"] == 1.0
    assert summary["citation_jaccard_avg"] == 1.0
    assert summary["retrieved_chunk_jaccard_avg"] == 1.0
    assert summary["passed"] is True


def test_summarize_group_fails_inconsistent_synthetic_responses():
    records = [
        _record(
            identified_flag_codes=[f"RF-{index}"],
            citation_chunk_ids=[f"chunk-{index}"],
            retrieved_chunk_ids=[f"chunk-{index}"],
        )
        for index in range(1, 5)
    ]

    summary = run_cqc_eval.summarize_group(_valid_group(), records)

    assert summary["passed"] is False
    assert summary["flag_jaccard_avg"] < run_cqc_eval.SET_CONSISTENCY_THRESHOLD


def test_evaluate_variant_extracts_response_fields_without_network(monkeypatch):
    def fake_post_query(base_url, query, timeout):
        assert (base_url, query, timeout) == ("http://example.test", "Query", 1.0)
        return 200, {
            "assessment": "possible",
            "refusal": {"refused": False},
            "identified_flags": [{"code": "RF-02"}, {"code": "RF-01"}],
            "citations": [{"chunk_id": "chunk-2"}, {"chunk_id": "chunk-1"}],
            "debug": {"retrieved_chunk_ids": ["chunk-2", "chunk-1"]},
        }

    monkeypatch.setattr(run_cqc_eval, "post_query", fake_post_query)

    record, unreachable = run_cqc_eval.evaluate_variant(
        "group", {"variant_id": "variant", "query": "Query"}, "http://example.test", 1.0
    )

    assert unreachable is False
    assert record["errors"] == []
    assert record["identified_flag_codes"] == ["RF-01", "RF-02"]
    assert record["citation_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert record["retrieved_chunk_ids"] == ["chunk-1", "chunk-2"]


def test_render_markdown_report_with_synthetic_group_results():
    passed = run_cqc_eval.summarize_group(_valid_group(), [_record() for _ in range(4)])
    failed = run_cqc_eval.summarize_group(
        _valid_group() | {"group_id": "failed-group"},
        [
            _record(
                identified_flag_codes=[f"RF-{index}"],
                citation_chunk_ids=[f"chunk-{index}"],
                retrieved_chunk_ids=[f"chunk-{index}"],
            )
            for index in range(1, 5)
        ],
    )

    report = run_cqc_eval.render_markdown_report(
        [passed, failed],
        "http://example.test",
        datetime(2026, 6, 15, 6, 30, tzinfo=timezone.utc),
    )

    assert report.startswith("# CQC-RAG Lite Evaluation Report\n")
    assert "**Timestamp:** 2026-06-15T06:30:00+00:00" in report
    assert "**Base URL:** `http://example.test`" in report
    assert "**Scenario groups passed:** 1 / 2" in report
    assert "## synthetic-group" in report
    assert "## failed-group" in report
    assert "**Identified flag consistency:**" in report
    assert "**Citation overlap summary:**" in report
    assert "**Retrieved chunk overlap summary:**" in report
    assert "**Pass/fail reason:** Failed consistency threshold:" in report
    assert "not a model-quality benchmark or full CQC-RAG reproduction" in report


def test_write_jsonl_preserves_line_oriented_output(tmp_path):
    records = [
        {"record_type": "variant", "query": "中文 query"},
        {"record_type": "group_summary", "passed": True},
    ]
    output = tmp_path / "results" / "cqc.jsonl"

    run_cqc_eval.write_jsonl(output, records)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records
    assert "中文 query" in lines[0]
