"""Classify observable issues in API smoke and CQC-RAG Lite result files."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_SMOKE = REPO_ROOT / "eval" / "results" / "api_smoke_latest.jsonl"
DEFAULT_CQC = REPO_ROOT / "eval" / "results" / "cqc_latest.jsonl"
DEFAULT_OUTPUT_MD = REPO_ROOT / "eval" / "reports" / "failure_diagnostics_latest.md"
DEFAULT_OUTPUT_JSONL = (
    REPO_ROOT / "eval" / "results" / "failure_diagnostics_latest.jsonl"
)
DEFAULT_OVERLAP_THRESHOLD = 0.50
SEVERITY_ORDER = ("HIGH", "MEDIUM", "LOW", "INFO")
INTERPRETATION_NOTE = [
    "This report is a local diagnostic artifact.",
    "It is not a model-quality benchmark.",
    "It does not reproduce the historical private-corpus retrieval benchmark.",
    "It does not modify retrieval, generation, gate, or pipeline behavior.",
    "Findings indicate where to inspect next, not final AML correctness.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose observable issues in existing evaluation JSONL outputs."
    )
    parser.add_argument("--api-smoke", type=Path, default=DEFAULT_API_SMOKE)
    parser.add_argument("--cqc", type=Path, default=DEFAULT_CQC)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 when any HIGH severity issue is found.",
    )
    return parser.parse_args()


def issue(
    source: str,
    severity: str,
    category: str,
    message: str,
    *,
    group_id: str | None = None,
    variant_id: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "severity": severity,
        "category": category,
        "group_id": group_id,
        "variant_id": variant_id,
        "message": message,
        "evidence": evidence or {},
    }


def read_jsonl(
    path: Path, source: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    input_status = {"source": source, "path": str(path), "found": path.exists()}
    if not path.exists():
        return [], input_status, [
            issue(
                "input",
                "MEDIUM",
                "SERVICE_UNREACHABLE_OR_MISSING_INPUT",
                f"{source} result file is unavailable: {path}",
                evidence={"path": str(path), "reason": "missing_input"},
            )
        ]

    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        input_status["readable"] = False
        issues.append(
            issue(
                "input",
                "HIGH",
                "SERVICE_UNREACHABLE_OR_MISSING_INPUT",
                f"{source} result file could not be read: {error}",
                evidence={"path": str(path), "reason": "read_error"},
            )
        )
        return records, input_status, issues

    input_status["readable"] = True
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not a JSON object")
            records.append(record)
        except (json.JSONDecodeError, ValueError) as error:
            issues.append(
                issue(
                    "input",
                    "HIGH",
                    "SERVICE_UNREACHABLE_OR_MISSING_INPUT",
                    f"Malformed {source} JSONL record at line {line_number}: {error}",
                    evidence={
                        "path": str(path),
                        "line_number": line_number,
                        "reason": "malformed_input",
                    },
                )
            )
    input_status["record_count"] = len(records)
    return records, input_status, issues


def _record_errors(record: dict[str, Any]) -> list[str]:
    errors = record.get("errors")
    if isinstance(errors, list):
        return [str(error) for error in errors if error]
    return [str(errors)] if errors else []


def _fallback_evidence(record: dict[str, Any]) -> tuple[bool, Any]:
    debug = record.get("debug")
    if not isinstance(debug, dict):
        debug = {}
    fallback_used = debug.get("fallback_used", record.get("fallback_used"))
    fallback_reason = debug.get("fallback_reason", record.get("fallback_reason"))
    return fallback_used is True or bool(fallback_reason), fallback_reason


def diagnose_api_smoke(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for record in records:
        record_id = record.get("id")
        errors = _record_errors(record)
        expected_status = record.get("expected_status", 200)
        status_code = record.get("status_code")
        failed = bool(errors) or record.get("passed") is False
        if status_code is not None and status_code != expected_status:
            failed = True
        if failed:
            issues.append(
                issue(
                    "api_smoke",
                    "HIGH",
                    "API_SMOKE_FAILURE",
                    f"API smoke case {record_id or '<unknown>'} failed observable checks.",
                    variant_id=str(record_id) if record_id is not None else None,
                    evidence={
                        "status_code": status_code,
                        "expected_status": expected_status,
                        "errors": errors,
                    },
                )
            )
        if any("request failed" in error.lower() for error in errors):
            issues.append(
                issue(
                    "api_smoke",
                    "HIGH",
                    "SERVICE_UNREACHABLE_OR_MISSING_INPUT",
                    f"API smoke case {record_id or '<unknown>'} records an unreachable service.",
                    variant_id=str(record_id) if record_id is not None else None,
                    evidence={"errors": errors},
                )
            )
        fallback_used, fallback_reason = _fallback_evidence(record)
        if fallback_used:
            issues.append(
                issue(
                    "api_smoke",
                    "MEDIUM",
                    "RETRIEVAL_FALLBACK",
                    f"API smoke case {record_id or '<unknown>'} used a recorded fallback.",
                    variant_id=str(record_id) if record_id is not None else None,
                    evidence={"fallback_reason": fallback_reason},
                )
            )
    return issues


def _metric_issue(
    summary: dict[str, Any],
    metric: str,
    category: str,
    label: str,
) -> dict[str, Any] | None:
    value = summary.get(metric)
    threshold = summary.get("set_consistency_threshold", DEFAULT_OVERLAP_THRESHOLD)
    if isinstance(value, (int, float)) and value < threshold:
        return issue(
            "cqc",
            "MEDIUM",
            category,
            f"CQC group {summary.get('group_id', '<unknown>')} has low {label}.",
            group_id=summary.get("group_id"),
            evidence={metric: value, "threshold": threshold},
        )
    return None


def diagnose_cqc(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    summaries = {
        record.get("group_id"): record
        for record in records
        if record.get("record_type") == "group_summary"
    }
    for summary in summaries.values():
        for metric, category, label in (
            ("citation_jaccard_avg", "LOW_CITATION_OVERLAP", "citation overlap"),
            (
                "retrieved_chunk_jaccard_avg",
                "RETRIEVED_CHUNK_INSTABILITY",
                "retrieved chunk overlap",
            ),
            ("flag_jaccard_avg", "IDENTIFIED_FLAG_INSTABILITY", "identified flag overlap"),
        ):
            metric_diagnostic = _metric_issue(summary, metric, category, label)
            if metric_diagnostic:
                issues.append(metric_diagnostic)

        for metric, category, label in (
            ("assessment_consistency", "ASSESSMENT_INSTABILITY", "assessment"),
            ("refusal_consistency", "REFUSAL_INSTABILITY", "refusal"),
        ):
            value = summary.get(metric)
            if isinstance(value, (int, float)) and value < 1.0:
                issues.append(
                    issue(
                        "cqc",
                        "HIGH",
                        category,
                        f"CQC group {summary.get('group_id', '<unknown>')} has unstable {label} outcomes.",
                        group_id=summary.get("group_id"),
                        evidence={metric: value, "expected": 1.0},
                    )
                )

        if summary.get("passed") is False:
            issues.append(
                issue(
                    "cqc",
                    "HIGH",
                    "EXPECTED_BEHAVIOR_MISMATCH",
                    f"CQC group {summary.get('group_id', '<unknown>')} did not pass its configured expectations.",
                    group_id=summary.get("group_id"),
                    evidence={
                        "expected_behavior": summary.get("expected_behavior"),
                        "passed": False,
                    },
                )
            )

    for record in records:
        if record.get("record_type") != "variant":
            continue
        group_id = record.get("group_id")
        variant_id = record.get("variant_id")
        summary = summaries.get(group_id, {})
        expected_behavior = summary.get("expected_behavior")
        errors = _record_errors(record)
        if any("request failed" in error.lower() for error in errors):
            issues.append(
                issue(
                    "cqc",
                    "HIGH",
                    "SERVICE_UNREACHABLE_OR_MISSING_INPUT",
                    f"CQC variant {variant_id or '<unknown>'} records an unreachable service.",
                    group_id=group_id,
                    variant_id=variant_id,
                    evidence={"errors": errors},
                )
            )

        fallback_used, fallback_reason = _fallback_evidence(record)
        if fallback_used:
            issues.append(
                issue(
                    "cqc",
                    "MEDIUM",
                    "RETRIEVAL_FALLBACK",
                    f"CQC variant {variant_id or '<unknown>'} used a recorded fallback.",
                    group_id=group_id,
                    variant_id=variant_id,
                    evidence={"fallback_reason": fallback_reason},
                )
            )

        refused = record.get("refusal_refused") is True or record.get("assessment") == "refuse"
        if refused:
            expected_refusal = expected_behavior == "stable_refuse"
            issues.append(
                issue(
                    "cqc",
                    "INFO" if expected_refusal else "HIGH",
                    "OUT_OF_SCOPE_REFUSAL",
                    (
                        f"CQC variant {variant_id or '<unknown>'} produced an "
                        f"{'expected' if expected_refusal else 'unexpected'} refusal."
                    ),
                    group_id=group_id,
                    variant_id=variant_id,
                    evidence={"expected_behavior": expected_behavior},
                )
            )

        citations = record.get("citation_chunk_ids")
        if (
            expected_behavior == "stable_possible"
            and record.get("assessment") == "unlikely"
            and isinstance(citations, list)
            and not citations
        ):
            issues.append(
                issue(
                    "cqc",
                    "MEDIUM",
                    "INSUFFICIENT_EVIDENCE",
                    f"CQC variant {variant_id or '<unknown>'} returned unlikely with no citations for a stable_possible group.",
                    group_id=group_id,
                    variant_id=variant_id,
                    evidence={"assessment": "unlikely", "citation_count": 0},
                )
            )
    return issues


def collect_diagnostics(
    api_smoke_path: Path, cqc_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    api_records, api_status, api_input_issues = read_jsonl(api_smoke_path, "api_smoke")
    cqc_records, cqc_status, cqc_input_issues = read_jsonl(cqc_path, "cqc")
    issues = (
        api_input_issues
        + cqc_input_issues
        + diagnose_api_smoke(api_records)
        + diagnose_cqc(cqc_records)
    )
    return issues, [api_status, cqc_status], api_records, cqc_records


def _count_table(counts: Counter[str], empty_label: str) -> list[str]:
    if not counts:
        return [f"- {empty_label}"]
    return [f"- **{name}:** {count}" for name, count in counts.items()]


def render_markdown_report(
    issues: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    api_records: list[dict[str, Any]],
    cqc_records: list[dict[str, Any]],
    timestamp: datetime | None = None,
) -> str:
    generated_at = timestamp or datetime.now(timezone.utc)
    severity_counts = Counter(item["severity"] for item in issues)
    category_counts = Counter(item["category"] for item in issues)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in issues:
        by_category[item["category"]].append(item)
        if item.get("group_id"):
            by_group[item["group_id"]].append(item)

    lines = [
        "# Failure Diagnostics Lite Report",
        "",
        f"- **Timestamp:** {generated_at.astimezone(timezone.utc).isoformat()}",
        f"- **Total issues:** {len(issues)}",
        "",
        "## Inputs",
        "",
    ]
    for input_status in inputs:
        state = "available" if input_status["found"] else "unavailable"
        count = input_status.get("record_count")
        suffix = f"; {count} records read" if count is not None else ""
        lines.append(
            f"- **{input_status['source']}:** {state} (`{input_status['path']}`{suffix})"
        )

    lines.extend(["", "## Issue Counts By Severity", ""])
    ordered_severity_counts = Counter(
        {
            severity: severity_counts[severity]
            for severity in SEVERITY_ORDER
            if severity_counts[severity]
        }
    )
    lines.extend(_count_table(ordered_severity_counts, "No issues detected."))
    lines.extend(["", "## Issue Counts By Category", ""])
    lines.extend(_count_table(Counter(dict(sorted(category_counts.items()))), "NO_ISSUE_DETECTED: 0"))

    lines.extend(["", "## Per-Category Summary", ""])
    if not by_category:
        lines.append("- **NO_ISSUE_DETECTED:** No observable diagnostic issues were found.")
    for category in sorted(by_category):
        lines.append(f"### {category}")
        lines.append("")
        for item in by_category[category]:
            lines.append(f"- **{item['severity']}:** {item['message']}")
        lines.append("")

    summaries = [
        record for record in cqc_records if record.get("record_type") == "group_summary"
    ]
    lines.extend(["## Per-Group CQC Diagnostics", ""])
    if not summaries:
        lines.append("- No CQC group summaries were available.")
    for summary in summaries:
        group_id = summary.get("group_id", "<unknown>")
        group_issues = by_group.get(group_id, [])
        lines.extend(
            [
                f"### {group_id}",
                "",
                f"- **Passed:** {summary.get('passed')}",
                f"- **Expected behavior:** {summary.get('expected_behavior', 'unavailable')}",
                f"- **Diagnostic issues:** {len(group_issues)}",
            ]
        )
        for item in group_issues:
            lines.append(f"- `{item['category']}` ({item['severity']}): {item['message']}")
        lines.append("")

    api_issues = [item for item in issues if item["source"] == "api_smoke"]
    lines.extend(["## API Smoke Diagnostics", ""])
    lines.append(f"- **Records available:** {len(api_records)}")
    if not api_issues:
        lines.append("- No API smoke diagnostic issues were found.")
    for item in api_issues:
        lines.append(f"- `{item['category']}` ({item['severity']}): {item['message']}")

    lines.extend(["", "## Interpretation Note", ""])
    lines.extend(f"- {note}" for note in INTERPRETATION_NOTE)
    return "\n".join(lines) + "\n"


def write_jsonl(path: Path, issues: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for item in issues:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_markdown_report(
    path: Path,
    issues: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    api_records: list[dict[str, Any]],
    cqc_records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_report(issues, inputs, api_records, cqc_records),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    issues, inputs, api_records, cqc_records = collect_diagnostics(
        args.api_smoke, args.cqc
    )
    write_jsonl(args.output_jsonl, issues)
    write_markdown_report(args.output_md, issues, inputs, api_records, cqc_records)

    severity_counts = Counter(item["severity"] for item in issues)
    print(
        "Failure Diagnostics Lite: "
        f"{len(issues)} issues "
        f"(HIGH={severity_counts['HIGH']}, MEDIUM={severity_counts['MEDIUM']}, "
        f"LOW={severity_counts['LOW']}, INFO={severity_counts['INFO']})"
    )
    print(f"Report written to {args.output_md}")
    return 1 if args.strict and severity_counts["HIGH"] else 0


if __name__ == "__main__":
    sys.exit(main())
