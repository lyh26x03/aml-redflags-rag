"""Tests for the multi-turn memory evaluation harness (network-free)."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import run_multiturn_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_PATH = REPO_ROOT / "eval" / "queries" / "multiturn_sessions_4.json"


def _session():
    return {
        "session_id": "sess",
        "description": "synthetic",
        "turns": [
            {
                "turn_id": "t1",
                "query": "scenario",
                "expected_route": "retrieve",
                "expect": {"min_active_flags": 1, "memory_updated": True},
            }
        ],
        "expect_memory": {"min_turns": 1},
    }


def _body(route="retrieve", **debug_overrides):
    debug = {
        "intent_route": route,
        "memory_used": False,
        "memory_updated": True,
        "referenced_previous_answer": False,
        "referenced_previous_evidence": False,
        "active_flags": ["RF-02"],
        "active_citation_count": 1,
    }
    debug.update(debug_overrides)
    return {
        "assessment": "possible",
        "refusal": {"refused": False},
        "identified_flags": [{"code": "RF-02"}],
        "citations": [{"chunk_id": "c0"}],
        "debug": debug,
    }


# --- input validation --------------------------------------------------------


def test_loads_committed_sessions():
    sessions = run_multiturn_eval.load_sessions(SESSIONS_PATH)
    assert len(sessions) == 4
    ids = [s["session_id"] for s in sessions]
    assert len(ids) == len(set(ids))
    routes = {
        turn["expected_route"] for s in sessions for turn in s["turns"]
    }
    assert routes <= run_multiturn_eval.VALID_ROUTES


@pytest.mark.parametrize(
    ("sessions", "message"),
    [
        ({}, "non-empty JSON array"),
        ([{"session_id": ""}], "non-empty session_id"),
        ([_session() | {"turns": []}], "turns must be a non-empty array"),
    ],
)
def test_malformed_sessions_fail_clearly(sessions, message):
    with pytest.raises(ValueError, match=message):
        run_multiturn_eval.validate_sessions(sessions)


def test_duplicate_session_ids_fail_clearly():
    with pytest.raises(ValueError, match="duplicate session_id: sess"):
        run_multiturn_eval.validate_sessions([_session(), copy.deepcopy(_session())])


def test_invalid_expected_route_fails_clearly():
    bad = _session()
    bad["turns"][0]["expected_route"] = "teleport"
    with pytest.raises(ValueError, match="invalid expected_route"):
        run_multiturn_eval.validate_sessions([bad])


# --- pure turn checks --------------------------------------------------------


def test_check_turn_passes_when_expectations_met():
    turn = _session()["turns"][0]
    record = run_multiturn_eval.check_turn(turn, _body())
    assert record["errors"] == []
    assert record["actual_route"] == "retrieve"
    assert record["active_flags"] == ["RF-02"]


def test_check_turn_flags_route_mismatch():
    turn = _session()["turns"][0]
    record = run_multiturn_eval.check_turn(turn, _body(route="clarify"))
    assert any("route" in error for error in record["errors"])


def test_check_turn_flags_missing_active_flags():
    turn = _session()["turns"][0]
    record = run_multiturn_eval.check_turn(turn, _body(active_flags=[]))
    assert any("min_active_flags" in error for error in record["errors"])


def test_check_turn_no_flags_expectation():
    turn = {
        "turn_id": "t",
        "query": "vague",
        "expected_route": "clarify",
        "expect": {"no_flags": True},
    }
    body = _body(route="clarify")
    body["identified_flags"] = [{"code": "RF-02"}]
    record = run_multiturn_eval.check_turn(turn, body)
    assert any("no_flags" in error for error in record["errors"])


# --- memory checks -----------------------------------------------------------


def test_check_memory_bounds():
    snapshot = {
        "turn_count": 1,
        "active_flags": [{"code": "RF-02"}],
        "active_citations": [{"chunk_id": "c0"}],
        "unresolved_questions": [],
    }
    assert run_multiturn_eval.check_memory({"min_turns": 1}, snapshot) == []
    assert run_multiturn_eval.check_memory({"max_active_flags": 0}, snapshot)
    assert run_multiturn_eval.check_memory({"min_unresolved": 1}, snapshot)


# --- summary + report --------------------------------------------------------


def test_summarize_session_pass_and_fail():
    session = _session()
    good = run_multiturn_eval.check_turn(session["turns"][0], _body())
    passed = run_multiturn_eval.summarize_session(session, [good], [], True)
    assert passed["passed"] is True

    bad = run_multiturn_eval.check_turn(session["turns"][0], _body(route="refuse"))
    failed = run_multiturn_eval.summarize_session(session, [bad], [], True)
    assert failed["passed"] is False


def test_summarize_session_fails_when_memory_unavailable():
    session = _session()
    good = run_multiturn_eval.check_turn(session["turns"][0], _body())
    summary = run_multiturn_eval.summarize_session(
        session, [good], ["memory snapshot unavailable"], False
    )
    assert summary["passed"] is False


def test_render_markdown_report():
    session = _session()
    good = run_multiturn_eval.check_turn(session["turns"][0], _body())
    summary = run_multiturn_eval.summarize_session(session, [good], [], True)
    report = run_multiturn_eval.render_markdown_report(
        [summary], "http://example.test", datetime(2026, 6, 16, tzinfo=timezone.utc)
    )
    assert report.startswith("# Multi-Turn Conversation Memory Evaluation Report\n")
    assert "**Sessions passed:** 1 / 1" in report
    assert "## sess" in report
    assert "not a model-quality benchmark" in report


def test_write_jsonl_roundtrip(tmp_path):
    records = [
        {"record_type": "turn", "query": "中文 query"},
        {"record_type": "session_summary", "passed": True},
    ]
    output = tmp_path / "results" / "multiturn.jsonl"
    run_multiturn_eval.write_jsonl(output, records)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records
    assert "中文 query" in lines[0]
