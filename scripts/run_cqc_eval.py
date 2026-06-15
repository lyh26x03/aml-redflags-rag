"""Run cross-query consistency evaluation against a running FastAPI service."""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "eval" / "queries" / "cqc_scenarios_5.json"
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "results" / "cqc_latest.jsonl"
EXPECTED_OUTCOMES = {
    "stable_possible": ("possible", False),
    "stable_unlikely": ("unlikely", False),
    "stable_refuse": ("refuse", True),
}
SET_CONSISTENCY_THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CQC-RAG lite against a running FastAPI service."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def post_query(base_url: str, query: str, timeout: float) -> tuple[int, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/query",
        data=json.dumps(
            {
                "query": query,
                "llm_mode": "mock",
                "retrieval_mode": "hybrid",
                "include_debug": True,
            }
        ).encode("utf-8"),
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


def _string_values(items: Any, key: str | None = None) -> list[str]:
    if not isinstance(items, list):
        return []
    values = []
    for item in items:
        value = item.get(key) if key and isinstance(item, dict) else item
        if isinstance(value, str):
            values.append(value)
    return sorted(set(values))


def evaluate_variant(
    group_id: str,
    variant: dict[str, str],
    base_url: str,
    timeout: float,
) -> tuple[dict[str, Any], bool]:
    errors: list[str] = []
    status_code: int | None = None
    body: Any = None
    service_unreachable = False

    try:
        status_code, body = post_query(base_url, variant["query"], timeout)
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        errors.append(f"request failed: {error}")
        service_unreachable = True

    if status_code != 200:
        errors.append(f"expected HTTP 200, got {status_code}")
    if not isinstance(body, dict):
        errors.append("response body is not a JSON object")
        body = {}

    refusal = body.get("refusal")
    debug = body.get("debug")
    record = {
        "record_type": "variant",
        "group_id": group_id,
        "variant_id": variant["variant_id"],
        "query": variant["query"],
        "status_code": status_code,
        "assessment": body.get("assessment"),
        "refusal_refused": (
            refusal.get("refused") if isinstance(refusal, dict) else None
        ),
        "identified_flag_codes": _string_values(body.get("identified_flags"), "code"),
        "citation_chunk_ids": _string_values(body.get("citations"), "chunk_id"),
        "retrieved_chunk_ids": (
            _string_values(debug.get("retrieved_chunk_ids"))
            if isinstance(debug, dict)
            else []
        ),
        "errors": errors,
    }
    return record, service_unreachable


def majority_ratio(values: list[Any]) -> float:
    if not values:
        return 0.0
    return max(Counter(values).values()) / len(values)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def average_pairwise_jaccard(records: list[dict[str, Any]], field: str) -> float:
    pairs = list(combinations((set(record[field]) for record in records), 2))
    if not pairs:
        return 1.0
    return sum(jaccard(left, right) for left, right in pairs) / len(pairs)


def summarize_group(
    group: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_behavior = group["expected_behavior"]
    expected_assessment, expected_refusal = EXPECTED_OUTCOMES[expected_behavior]
    assessment_consistency = majority_ratio(
        [record["assessment"] for record in records]
    )
    refusal_consistency = majority_ratio(
        [record["refusal_refused"] for record in records]
    )
    flag_jaccard_avg = average_pairwise_jaccard(records, "identified_flag_codes")
    citation_jaccard_avg = average_pairwise_jaccard(records, "citation_chunk_ids")
    retrieved_chunk_jaccard_avg = average_pairwise_jaccard(
        records, "retrieved_chunk_ids"
    )
    outcomes_match = all(
        record["assessment"] == expected_assessment
        and record["refusal_refused"] is expected_refusal
        for record in records
    )
    sets_consistent = all(
        value >= SET_CONSISTENCY_THRESHOLD
        for value in (
            flag_jaccard_avg,
            citation_jaccard_avg,
            retrieved_chunk_jaccard_avg,
        )
    )
    passed = (
        bool(records)
        and not any(record["errors"] for record in records)
        and outcomes_match
        and sets_consistent
    )
    return {
        "record_type": "group_summary",
        "group_id": group["group_id"],
        "description": group["description"],
        "expected_behavior": expected_behavior,
        "variant_count": len(records),
        "assessment_consistency": round(assessment_consistency, 4),
        "refusal_consistency": round(refusal_consistency, 4),
        "flag_jaccard_avg": round(flag_jaccard_avg, 4),
        "citation_jaccard_avg": round(citation_jaccard_avg, 4),
        "retrieved_chunk_jaccard_avg": round(retrieved_chunk_jaccard_avg, 4),
        "set_consistency_threshold": SET_CONSISTENCY_THRESHOLD,
        "passed": passed,
    }


def validate_groups(groups: Any) -> list[dict[str, Any]]:
    if not isinstance(groups, list):
        raise ValueError("input must be a JSON array of scenario groups")
    group_ids: set[str] = set()
    variant_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("each scenario group must be a JSON object")
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("each scenario group must have a non-empty group_id")
        if group_id in group_ids:
            raise ValueError(f"duplicate group_id: {group_id}")
        group_ids.add(group_id)
        if (
            not isinstance(group.get("description"), str)
            or not group["description"].strip()
        ):
            raise ValueError(f"{group_id}: description must be a non-empty string")
        if group.get("expected_behavior") not in EXPECTED_OUTCOMES:
            raise ValueError(f"{group_id}: invalid expected_behavior")
        variants = group.get("variants")
        if not isinstance(variants, list) or not 3 <= len(variants) <= 5:
            raise ValueError(f"{group_id}: variants must contain 3 to 5 items")
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError(f"{group_id}: each variant must be a JSON object")
            variant_id = variant.get("variant_id")
            query = variant.get("query")
            if not isinstance(variant_id, str) or not variant_id.strip():
                raise ValueError(f"{group_id}: each variant needs a non-empty variant_id")
            if variant_id in variant_ids:
                raise ValueError(f"duplicate variant_id: {variant_id}")
            variant_ids.add(variant_id)
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{variant_id}: query must be a non-empty string")
    return groups


def load_groups(path: Path) -> list[dict[str, Any]]:
    return validate_groups(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    args = parse_args()
    try:
        groups = load_groups(args.input)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"CQC-RAG lite input error: {error}", file=sys.stderr)
        return 1

    output_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    service_unreachable = False
    for group in groups:
        variant_records = []
        for variant in group["variants"]:
            record, unreachable = evaluate_variant(
                group["group_id"], variant, args.base_url, args.timeout
            )
            variant_records.append(record)
            output_records.append(record)
            service_unreachable = service_unreachable or unreachable
        summary = summarize_group(group, variant_records)
        summaries.append(summary)
        output_records.append(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in output_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    passed = sum(summary["passed"] for summary in summaries)
    print(f"CQC-RAG lite: {passed} / {len(summaries)} groups passed")
    for summary in summaries:
        print(
            f"- {summary['group_id']}: "
            f"assessment={summary['assessment_consistency']:.4f}, "
            f"flags={summary['flag_jaccard_avg']:.4f}, "
            f"citations={summary['citation_jaccard_avg']:.4f}, "
            f"retrieved_chunks={summary['retrieved_chunk_jaccard_avg']:.4f}, "
            f"passed={summary['passed']}"
        )
    return 0 if passed == len(summaries) and not service_unreachable else 1


if __name__ == "__main__":
    sys.exit(main())
