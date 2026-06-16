"""Deterministic multi-turn memory evaluator for a running FastAPI service.

This harness drives the structured-conversation-memory routes through a few
fixed multi-turn sessions and checks the observable contract:

- Session A: AML scenario -> follow-up recalls the prior flags
- Session B: vague first-turn query -> clarification
- Session C: out-of-scope query -> refusal that does not pollute memory
- Session D: AML scenario -> follow-up asks for the previous citations

It is deterministic (mock generation, BM25 retrieval), calls only the service
API, and writes:

- ``eval/results/multiturn_latest.jsonl``  (per-turn + per-session records)
- ``eval/reports/multiturn_latest.md``     (human-readable report)

Both outputs are gitignored. This is a behavior smoke harness, not a
model-quality benchmark.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "eval" / "queries" / "multiturn_sessions_4.json"
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "results" / "multiturn_latest.jsonl"
DEFAULT_REPORT_MD = REPO_ROOT / "eval" / "reports" / "multiturn_latest.md"

VALID_ROUTES = {
    "retrieve",
    "refuse",
    "ask_clarifying_question",
    "answer_from_history",
    "retrieve_with_memory",
}

# The five fine-grained routes collapse onto three reviewer-facing outcomes.
# Kept as a local mirror of rag_core.intent_router.route_family so this harness
# stays stdlib-only (it talks to the service over HTTP, not via imports).
ROUTE_FAMILY = {
    "retrieve": "retrieve",
    "retrieve_with_memory": "retrieve",
    "refuse": "refuse",
    "answer_from_history": "no_retrieval_response",
    "ask_clarifying_question": "no_retrieval_response",
}
VALID_ROUTE_FAMILIES = set(ROUTE_FAMILY.values())


# --- input validation ---------------------------------------------------------


def validate_sessions(sessions: Any) -> List[Dict[str, Any]]:
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("input must be a non-empty JSON array of sessions")
    session_ids: set = set()
    turn_ids: set = set()
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError("each session must be a JSON object")
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("each session needs a non-empty session_id")
        if session_id in session_ids:
            raise ValueError(f"duplicate session_id: {session_id}")
        session_ids.add(session_id)
        if not isinstance(session.get("description"), str) or not session[
            "description"
        ].strip():
            raise ValueError(f"{session_id}: description must be a non-empty string")
        turns = session.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{session_id}: turns must be a non-empty array")
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError(f"{session_id}: each turn must be a JSON object")
            turn_id = turn.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id.strip():
                raise ValueError(f"{session_id}: each turn needs a non-empty turn_id")
            if turn_id in turn_ids:
                raise ValueError(f"duplicate turn_id: {turn_id}")
            turn_ids.add(turn_id)
            if not isinstance(turn.get("query"), str) or not turn["query"].strip():
                raise ValueError(f"{turn_id}: query must be a non-empty string")
            if turn.get("expected_route") not in VALID_ROUTES:
                raise ValueError(f"{turn_id}: invalid expected_route")
            if (
                "expected_family" in turn
                and turn["expected_family"] not in VALID_ROUTE_FAMILIES
            ):
                raise ValueError(f"{turn_id}: invalid expected_family")
            if not isinstance(turn.get("expect", {}), dict):
                raise ValueError(f"{turn_id}: expect must be a JSON object")
        if not isinstance(session.get("expect_memory", {}), dict):
            raise ValueError(f"{session_id}: expect_memory must be a JSON object")
    return sessions


def load_sessions(path: Path) -> List[Dict[str, Any]]:
    return validate_sessions(json.loads(path.read_text(encoding="utf-8")))


# --- HTTP helpers -------------------------------------------------------------


def post_query(
    base_url: str, payload: Dict[str, Any], timeout: float
) -> Tuple[int, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send(request, timeout)


def get_memory(base_url: str, session_id: str, timeout: float) -> Tuple[int, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/sessions/{session_id}/memory",
        method="GET",
    )
    return _send(request, timeout)


def _send(request: Request, timeout: float) -> Tuple[int, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raw = error.read().decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw
        return error.code, body


# --- pure checks (network-free, unit-testable) --------------------------------


def check_turn(turn: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    """Compare one turn's response body against its expectations."""
    expect = turn.get("expect", {})
    debug = body.get("debug") if isinstance(body.get("debug"), dict) else {}
    refusal = body.get("refusal") if isinstance(body.get("refusal"), dict) else {}
    flags = body.get("identified_flags") if isinstance(
        body.get("identified_flags"), list
    ) else []
    active_flags = debug.get("active_flags") if isinstance(
        debug.get("active_flags"), list
    ) else []
    active_citation_count = debug.get("active_citation_count") or 0

    errors: List[str] = []
    actual_route = debug.get("intent_route")
    if actual_route != turn["expected_route"]:
        errors.append(
            f"route: expected {turn['expected_route']}, got {actual_route}"
        )

    # High-level outcome: prefer the service-reported family, fall back to the
    # local mirror so the check works even against older debug payloads.
    actual_family = debug.get("route_family") or ROUTE_FAMILY.get(actual_route)
    expected_family = turn.get("expected_family") or ROUTE_FAMILY.get(
        turn["expected_route"]
    )
    if expected_family is not None and actual_family != expected_family:
        errors.append(
            f"route_family: expected {expected_family}, got {actual_family}"
        )

    bool_fields = (
        "memory_used",
        "memory_updated",
        "referenced_previous_answer",
        "referenced_previous_evidence",
    )
    for key in bool_fields:
        if key in expect and bool(debug.get(key)) != bool(expect[key]):
            errors.append(f"{key}: expected {expect[key]}, got {debug.get(key)}")

    if "assessment" in expect and body.get("assessment") != expect["assessment"]:
        errors.append(
            f"assessment: expected {expect['assessment']}, got {body.get('assessment')}"
        )
    if "refused" in expect and bool(refusal.get("refused")) != bool(expect["refused"]):
        errors.append(
            f"refused: expected {expect['refused']}, got {refusal.get('refused')}"
        )
    if expect.get("no_flags") and flags:
        errors.append(f"no_flags: expected 0 flags, got {len(flags)}")
    if "min_active_flags" in expect and len(active_flags) < expect["min_active_flags"]:
        errors.append(
            f"min_active_flags: expected >= {expect['min_active_flags']}, "
            f"got {len(active_flags)}"
        )
    if (
        "min_active_citations" in expect
        and active_citation_count < expect["min_active_citations"]
    ):
        errors.append(
            f"min_active_citations: expected >= {expect['min_active_citations']}, "
            f"got {active_citation_count}"
        )

    return {
        "record_type": "turn",
        "session_id": None,  # filled in by the caller
        "turn_id": turn["turn_id"],
        "query": turn["query"],
        "expected_route": turn["expected_route"],
        "actual_route": actual_route,
        "expected_family": expected_family,
        "actual_family": actual_family,
        "assessment": body.get("assessment"),
        "refused": bool(refusal.get("refused")),
        "memory_used": bool(debug.get("memory_used")),
        "memory_updated": bool(debug.get("memory_updated")),
        "referenced_previous_answer": bool(debug.get("referenced_previous_answer")),
        "referenced_previous_evidence": bool(debug.get("referenced_previous_evidence")),
        "active_flags": list(active_flags),
        "active_citation_count": active_citation_count,
        "errors": errors,
    }


def check_memory(expect_memory: Dict[str, Any], snapshot: Dict[str, Any]) -> List[str]:
    """Compare the end-of-session memory snapshot against its expectations."""
    errors: List[str] = []
    turn_count = snapshot.get("turn_count", 0)
    active_flags = snapshot.get("active_flags") or []
    active_citations = snapshot.get("active_citations") or []
    unresolved = snapshot.get("unresolved_questions") or []

    bounds = (
        ("min_turns", turn_count, "turn_count", ">="),
        ("max_turns", turn_count, "turn_count", "<="),
        ("min_active_flags", len(active_flags), "active_flags", ">="),
        ("max_active_flags", len(active_flags), "active_flags", "<="),
        ("min_active_citations", len(active_citations), "active_citations", ">="),
        ("min_unresolved", len(unresolved), "unresolved_questions", ">="),
    )
    for key, actual, label, op in bounds:
        if key not in expect_memory:
            continue
        expected = expect_memory[key]
        ok = actual >= expected if op == ">=" else actual <= expected
        if not ok:
            errors.append(f"{label} {op} {expected} failed (got {actual})")
    return errors


def summarize_session(
    session: Dict[str, Any],
    turn_records: List[Dict[str, Any]],
    memory_errors: List[str],
    memory_available: bool,
) -> Dict[str, Any]:
    turn_errors = sum(len(record["errors"]) for record in turn_records)
    passed = (
        bool(turn_records)
        and turn_errors == 0
        and not memory_errors
        and memory_available
    )
    return {
        "record_type": "session_summary",
        "session_id": session["session_id"],
        "description": session["description"],
        "turn_count": len(turn_records),
        "turn_errors": turn_errors,
        "memory_available": memory_available,
        "memory_errors": memory_errors,
        "passed": passed,
    }


# --- session execution (network) ----------------------------------------------


def run_session(
    session: Dict[str, Any], base_url: str, timeout: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    """Run one session's turns and snapshot its memory. Returns
    (turn_records, session_summary, service_unreachable)."""
    turn_records: List[Dict[str, Any]] = []
    unreachable = False
    for index, turn in enumerate(session["turns"]):
        payload = {
            "query": turn["query"],
            "llm_mode": "mock",
            "retrieval_mode": "bm25",
            "include_debug": True,
            "session_id": session["session_id"],
            "use_memory": True,
        }
        if index == 0:
            # Reset first so re-runs start from a clean, deterministic state.
            payload["reset_memory"] = True
        try:
            status, body = post_query(base_url, payload, timeout)
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            unreachable = True
            record = check_turn(turn, {})
            record["session_id"] = session["session_id"]
            record["errors"].append(f"request failed: {error}")
            turn_records.append(record)
            continue
        if status != 200 or not isinstance(body, dict):
            record = check_turn(turn, body if isinstance(body, dict) else {})
            record["session_id"] = session["session_id"]
            record["errors"].append(f"expected HTTP 200, got {status}")
            turn_records.append(record)
            continue
        record = check_turn(turn, body)
        record["session_id"] = session["session_id"]
        turn_records.append(record)

    snapshot: Dict[str, Any] = {}
    memory_available = False
    try:
        status, body = get_memory(base_url, session["session_id"], timeout)
        if status == 200 and isinstance(body, dict):
            snapshot = body
            memory_available = True
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        unreachable = True
        snapshot = {"error": str(error)}

    memory_errors = (
        check_memory(session.get("expect_memory", {}), snapshot)
        if memory_available
        else ["memory snapshot unavailable"]
    )
    summary = summarize_session(
        session, turn_records, memory_errors, memory_available
    )
    return turn_records, summary, unreachable


# --- reporting ----------------------------------------------------------------


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_markdown_report(
    summaries: List[Dict[str, Any]],
    base_url: str,
    timestamp: Optional[datetime] = None,
) -> str:
    generated_at = timestamp or datetime.now(timezone.utc)
    passed = sum(summary["passed"] for summary in summaries)
    lines = [
        "# Multi-Turn Conversation Memory Evaluation Report",
        "",
        f"- **Timestamp:** {generated_at.astimezone(timezone.utc).isoformat()}",
        f"- **Base URL:** `{base_url}`",
        f"- **Sessions passed:** {passed} / {len(summaries)}",
        "",
        "> This is a deterministic multi-turn behavior smoke harness for the "
        "structured conversation memory routes, not a model-quality benchmark.",
        "",
        "## Session Status",
        "",
        "| Session | Status | Turns | Turn errors | Memory checks |",
        "|---|---|---:|---:|---|",
    ]
    for summary in summaries:
        status = "PASS" if summary["passed"] else "FAIL"
        memory_note = (
            "ok" if not summary["memory_errors"] else "; ".join(summary["memory_errors"])
        )
        lines.append(
            f"| `{summary['session_id']}` | **{status}** | {summary['turn_count']} | "
            f"{summary['turn_errors']} | {memory_note} |"
        )

    for summary in summaries:
        status = "PASS" if summary["passed"] else "FAIL"
        lines.extend(
            [
                "",
                f"## {summary['session_id']}",
                "",
                f"- **Status:** {status}",
                f"- **Description:** {summary['description']}",
                f"- **Turns evaluated:** {summary['turn_count']}",
                f"- **Turn errors:** {summary['turn_errors']}",
                f"- **Memory available:** {summary['memory_available']}",
                "- **Memory check result:** "
                + ("passed" if not summary["memory_errors"] else "; ".join(summary["memory_errors"])),
            ]
        )
    return "\n".join(lines) + "\n"


def write_markdown_report(
    path: Path,
    summaries: List[Dict[str, Any]],
    base_url: str,
    timestamp: Optional[datetime] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_report(summaries, base_url, timestamp), encoding="utf-8"
    )


# --- CLI ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-turn memory evaluator against a running service."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sessions = load_sessions(args.input)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"multi-turn eval input error: {error}", file=sys.stderr)
        return 1

    records: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    service_unreachable = False
    for session in sessions:
        turn_records, summary, unreachable = run_session(
            session, args.base_url, args.timeout
        )
        records.extend(turn_records)
        records.append(summary)
        summaries.append(summary)
        service_unreachable = service_unreachable or unreachable

    write_jsonl(args.output, records)
    write_markdown_report(args.report_md, summaries, args.base_url)

    passed = sum(summary["passed"] for summary in summaries)
    print(f"multi-turn eval: {passed} / {len(summaries)} sessions passed")
    for summary in summaries:
        print(
            f"- {summary['session_id']}: passed={summary['passed']}, "
            f"turn_errors={summary['turn_errors']}, "
            f"memory_errors={len(summary['memory_errors'])}"
        )
    return 0 if passed == len(summaries) and not service_unreachable else 1


if __name__ == "__main__":
    sys.exit(main())
