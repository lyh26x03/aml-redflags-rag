"""Run contract smoke evaluation cases against a running FastAPI service."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "eval" / "queries" / "api_smoke_8.json"
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "results" / "api_smoke_latest.jsonl"
RESPONSE_FIELDS = {
    "answer",
    "assessment",
    "identified_flags",
    "citations",
    "refusal",
    "debug",
}
REQUEST_FIELDS = {"query", "retrieval_mode", "llm_mode", "include_debug"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run API contract smoke cases against a running service."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def post_query(base_url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raw_body = error.read().decode("utf-8")
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            body = raw_body
        return error.code, body


def evaluate_case(case: dict[str, Any], base_url: str, timeout: float) -> dict[str, Any]:
    errors: list[str] = []
    status_code: int | None = None
    body: Any = None
    expected_status = case["expected_http_status"]

    try:
        payload = {field: case[field] for field in REQUEST_FIELDS}
        status_code, body = post_query(base_url, payload, timeout)
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        errors.append(f"request failed: {error}")

    if status_code != expected_status:
        errors.append(f"expected HTTP {expected_status}, got {status_code}")

    if not isinstance(body, dict):
        errors.append("response body is not a JSON object")
        body = {}

    missing_fields = sorted(RESPONSE_FIELDS - body.keys())
    if missing_fields:
        errors.append(f"missing response fields: {', '.join(missing_fields)}")

    assessment = body.get("assessment")
    expected_assessment = case.get("expected_assessment")
    if expected_assessment is not None and assessment != expected_assessment:
        errors.append(
            f"expected assessment {expected_assessment!r}, got {assessment!r}"
        )

    refusal = body.get("refusal")
    refused = refusal.get("refused") if isinstance(refusal, dict) else None
    if refused is not case["expect_refusal"]:
        errors.append(
            f"expected refusal.refused={case['expect_refusal']}, got {refused!r}"
        )

    citations = body.get("citations")
    citations_count = len(citations) if isinstance(citations, list) else 0
    has_citations = citations_count > 0
    if has_citations is not case["expect_citations"]:
        errors.append(
            f"expected citations={case['expect_citations']}, got {has_citations}"
        )

    debug = body.get("debug")
    if case["include_debug"] and debug is None:
        errors.append("expected debug to be non-null")
    if not case["include_debug"] and debug is not None:
        errors.append("expected debug to be null")

    return {
        "id": case["id"],
        "passed": not errors,
        "status_code": status_code,
        "expected_status": expected_status,
        "assessment": assessment,
        "refusal": refusal,
        "citations_count": citations_count,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    cases = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate_case(case, args.base_url, args.timeout)
        for case in cases
    ]
    with args.output.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    passed = sum(result["passed"] for result in results)
    print(f"API smoke eval: {passed} / {len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
