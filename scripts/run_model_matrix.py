"""Run a provider/mode behavior smoke matrix against the FastAPI service."""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = REPO_ROOT / "eval" / "queries" / "model_matrix_queries_6.json"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "eval" / "results" / "model_matrix_latest.jsonl"
DEFAULT_OUTPUT_MD = REPO_ROOT / "eval" / "reports" / "model_matrix_latest.md"
QUERY_FIELDS = {"query", "retrieval_mode", "llm_mode", "include_debug"}
INTERPRETATION_NOTE = [
    "This is a provider/mode behavior smoke matrix, not a model-quality benchmark.",
    "Results depend on the currently running service configuration and API keys.",
    "Mock mode is deterministic and remains the default reviewer path.",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run provider/mode matrix checks against a running FastAPI service."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--modes",
        default="mock",
        help="Comma-separated llm_mode values to request. Default: mock",
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--corpus-label")
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="Also write dated archive copies alongside the latest outputs.",
    )
    return parser.parse_args(argv)


def parse_modes(raw_modes: str) -> list[str]:
    modes: list[str] = []
    for item in raw_modes.split(","):
        mode = item.strip()
        if mode and mode not in modes:
            modes.append(mode)
    return modes or ["mock"]


def validate_queries(queries: Any) -> list[dict[str, str]]:
    if not isinstance(queries, list):
        raise ValueError("query file must contain a JSON array")
    validated: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in queries:
        if not isinstance(item, dict):
            raise ValueError("each query entry must be a JSON object")
        query_id = item.get("query_id")
        query = item.get("query")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("each query entry needs a non-empty query_id")
        if query_id in seen_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{query_id}: query must be a non-empty string")
        seen_ids.add(query_id)
        validated.append({"query_id": query_id, "query": query.strip()})
    return validated


def load_queries(path: Path) -> list[dict[str, str]]:
    return validate_queries(json.loads(path.read_text(encoding="utf-8")))


def _filename_token(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = [
        char.lower()
        if char.isalnum() or char in {"-", "_"}
        else "-"
        for char in value.strip()
    ]
    token = "".join(cleaned).strip("-_")
    while "--" in token:
        token = token.replace("--", "-")
    return token or fallback


def resolve_snapshot_paths(
    out_jsonl: Path,
    out_md: Path,
    *,
    timestamp: datetime,
    requested_modes: list[str],
    corpus_label: str | None = None,
    corpus_profile: str | None = None,
) -> tuple[Path, Path]:
    date_token = timestamp.strftime("%Y%m%d")
    corpus_token = _filename_token(
        corpus_label or corpus_profile,
        fallback="unknown",
    )
    modes_token = _filename_token("-".join(requested_modes), fallback="mock")
    base_stem = f"model_matrix_{date_token}_{corpus_token}_{modes_token}"
    json_archive_dir = out_jsonl.parent / "archive"
    md_archive_dir = out_md.parent / "archive"

    revision = 1
    while True:
        suffix = "" if revision == 1 else f"_r{revision}"
        jsonl_path = json_archive_dir / f"{base_stem}{suffix}.jsonl"
        md_path = md_archive_dir / f"{base_stem}{suffix}.md"
        if not jsonl_path.exists() and not md_path.exists():
            return jsonl_path, md_path
        revision += 1


def _decode_json(raw_body: str) -> Any:
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body


def fetch_json(
    base_url: str,
    path: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=(json.dumps(payload).encode("utf-8") if payload is not None else None),
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, _decode_json(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, _decode_json(error.read().decode("utf-8"))


def fetch_service_metadata(base_url: str, timeout: float) -> dict[str, Any]:
    metadata = {
        "health": None,
        "sources": None,
        "corpus_profile": None,
        "total_chunks": None,
    }
    for path, key in (("/health", "health"), ("/sources", "sources")):
        try:
            status_code, body = fetch_json(base_url, path, timeout)
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            continue
        if status_code == 200 and isinstance(body, dict):
            metadata[key] = body

    health = metadata["health"] or {}
    sources = metadata["sources"] or {}
    metadata["corpus_profile"] = (
        sources.get("corpus_profile") or health.get("corpus_profile")
    )
    metadata["total_chunks"] = sources.get("total_chunks") or health.get("chunk_count")
    return metadata


def sanitize_error_message(message: Any, max_length: int = 240) -> str | None:
    if message is None:
        return None
    text = " ".join(str(message).split())
    if not text:
        return None
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _detail_message(body: Any) -> str | None:
    if not isinstance(body, dict):
        return sanitize_error_message(body)
    detail = body.get("detail")
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(part) for part in item.get("loc", []))
                msg = item.get("msg")
                if msg and loc:
                    parts.append(f"{loc}: {msg}")
                elif msg:
                    parts.append(str(msg))
        if parts:
            return sanitize_error_message("; ".join(parts))
    return sanitize_error_message(body.get("message") or body.get("error") or body)


def is_unsupported_llm_mode(status_code: int | None, body: Any) -> bool:
    if status_code != 422 or not isinstance(body, dict):
        return False
    detail = body.get("detail")
    if not isinstance(detail, list):
        return False
    for item in detail:
        if not isinstance(item, dict):
            continue
        loc = item.get("loc", [])
        if isinstance(loc, list) and "llm_mode" in {str(part) for part in loc}:
            return True
    return False


def classify_http_outcome(
    status_code: int | None, body: Any
) -> tuple[str, str | None, str | None]:
    if status_code == 200:
        return "ok", None, None
    if is_unsupported_llm_mode(status_code, body):
        return "unsupported", "unsupported_llm_mode", _detail_message(body)
    if status_code == 503:
        return (
            "service_error",
            str(body.get("error", "http_503")) if isinstance(body, dict) else "http_503",
            _detail_message(body),
        )
    return (
        "error",
        f"http_{status_code}" if status_code is not None else None,
        _detail_message(body),
    )


def _string_values(items: Any, key: str | None = None) -> list[str]:
    if not isinstance(items, list):
        return []
    values = []
    for item in items:
        value = item.get(key) if key and isinstance(item, dict) else item
        if isinstance(value, str) and value:
            values.append(value)
    return sorted(set(values))


def evaluate_query_mode(
    run_id: str,
    timestamp_utc: str,
    base_url: str,
    mode: str,
    query_case: dict[str, str],
    timeout: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "query": query_case["query"],
        "retrieval_mode": "hybrid",
        "llm_mode": mode,
        "include_debug": True,
    }
    status_code: int | None = None
    body: Any = None
    latency_ms: int | None = None
    try:
        started = time.perf_counter()
        status_code, body = fetch_json(base_url, "/query", timeout, payload)
        latency_ms = int(round((time.perf_counter() - started) * 1000))
        status, error_type, error_message = classify_http_outcome(status_code, body)
    except TimeoutError as error:
        status = "error"
        error_type = "timeout"
        error_message = sanitize_error_message(error)
    except (URLError, OSError) as error:
        status = "error"
        error_type = type(error).__name__
        error_message = sanitize_error_message(error)
    except json.JSONDecodeError as error:
        status = "error"
        error_type = "json_decode_error"
        error_message = sanitize_error_message(error)

    response = body if isinstance(body, dict) else {}
    debug = response.get("debug") if isinstance(response.get("debug"), dict) else {}
    fallback_used = debug.get("fallback_used") is True
    fallback_reason = sanitize_error_message(debug.get("fallback_reason"))

    return {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "query_id": query_case["query_id"],
        "query": query_case["query"],
        "llm_mode": mode,
        "status": status,
        "http_status": status_code,
        "latency_ms": latency_ms,
        "assessment": response.get("assessment"),
        "identified_flag_codes": _string_values(response.get("identified_flags"), "code"),
        "citation_count": len(response.get("citations", []))
        if isinstance(response.get("citations"), list)
        else 0,
        "retrieved_chunk_ids": _string_values(debug.get("retrieved_chunk_ids")),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "parse_success": response.get("parse_success")
        if isinstance(response.get("parse_success"), bool)
        else None,
        "model": None,
        "provider": debug.get("llm_mode") if isinstance(debug.get("llm_mode"), str) else mode,
        "corpus_profile": metadata.get("corpus_profile"),
        "total_chunks": metadata.get("total_chunks"),
        "error_type": error_type,
        "error_message": error_message,
    }


def summarize_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_mode[str(record["llm_mode"])].append(record)

    summaries = []
    for mode, mode_records in by_mode.items():
        latencies = [
            record["latency_ms"]
            for record in mode_records
            if isinstance(record.get("latency_ms"), int)
        ]
        summaries.append(
            {
                "llm_mode": mode,
                "total": len(mode_records),
                "ok": sum(record["status"] == "ok" for record in mode_records),
                "unsupported": sum(
                    record["status"] == "unsupported" for record in mode_records
                ),
                "errors": sum(
                    record["status"] in {"error", "service_error"}
                    for record in mode_records
                ),
                "fallback_count": sum(
                    record.get("fallback_used") is True for record in mode_records
                ),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1)
                if latencies
                else None,
            }
        )
    return sorted(summaries, key=lambda item: item["llm_mode"])


def render_markdown_report(
    records: list[dict[str, Any]],
    *,
    run_id: str,
    timestamp_utc: str,
    base_url: str,
    requested_modes: list[str],
    metadata: dict[str, Any],
    corpus_label: str | None = None,
) -> str:
    summaries = summarize_results(records)
    lines = [
        "# Model Matrix Runner Report",
        "",
        f"- **Run ID:** `{run_id}`",
        f"- **Timestamp UTC:** {timestamp_utc}",
        f"- **Base URL:** `{base_url}`",
        f"- **Modes requested:** {', '.join(requested_modes)}",
        f"- **Corpus profile:** {metadata.get('corpus_profile') or 'unavailable'}",
        f"- **Total chunks:** {metadata.get('total_chunks') if metadata.get('total_chunks') is not None else 'unavailable'}",
    ]
    if corpus_label:
        lines.append(f"- **Corpus label:** {corpus_label}")

    lines.extend(
        [
            "",
            "## Summary By Mode",
            "",
            "| Mode | Total | OK | Unsupported | Errors | Fallbacks | Avg latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        avg_latency = (
            f"{summary['avg_latency_ms']:.1f}"
            if isinstance(summary["avg_latency_ms"], (int, float))
            else "-"
        )
        lines.append(
            f"| `{summary['llm_mode']}` | {summary['total']} | {summary['ok']} | "
            f"{summary['unsupported']} | {summary['errors']} | "
            f"{summary['fallback_count']} | {avg_latency} |"
        )

    lines.extend(
        [
            "",
            "## Per-Query Comparison",
            "",
            "| Query ID | Mode | Status | Assessment | Flags | Citation count | Fallback used |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for record in records:
        flags = ", ".join(record["identified_flag_codes"]) or "-"
        assessment = record["assessment"] or "-"
        lines.append(
            f"| `{record['query_id']}` | `{record['llm_mode']}` | {record['status']} | "
            f"{assessment} | {flags} | {record['citation_count']} | "
            f"{str(record['fallback_used']).lower()} |"
        )

    lines.extend(["", "## Interpretation Note", ""])
    lines.extend(f"- {note}" for note in INTERPRETATION_NOTE)
    return "\n".join(lines) + "\n"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown_report(
    path: Path,
    records: list[dict[str, Any]],
    *,
    run_id: str,
    timestamp_utc: str,
    base_url: str,
    requested_modes: list[str],
    metadata: dict[str, Any],
    corpus_label: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_report(
            records,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            base_url=base_url,
            requested_modes=requested_modes,
            metadata=metadata,
            corpus_label=corpus_label,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queries = load_queries(args.queries)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Model matrix input error: {error}", file=sys.stderr)
        return 1

    requested_modes = parse_modes(args.modes)
    timestamp = datetime.now(timezone.utc)
    timestamp_utc = timestamp.isoformat()
    run_id = timestamp.strftime("model-matrix-%Y%m%dT%H%M%SZ")
    metadata = fetch_service_metadata(args.base_url, args.timeout)

    records = [
        evaluate_query_mode(
            run_id,
            timestamp_utc,
            args.base_url,
            mode,
            query_case,
            args.timeout,
            metadata,
        )
        for mode in requested_modes
        for query_case in queries
    ]

    write_jsonl(args.out_jsonl, records)
    write_markdown_report(
        args.out_md,
        records,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        base_url=args.base_url,
        requested_modes=requested_modes,
        metadata=metadata,
        corpus_label=args.corpus_label,
    )

    snapshot_paths: tuple[Path, Path] | None = None
    if args.save_snapshot:
        snapshot_paths = resolve_snapshot_paths(
            args.out_jsonl,
            args.out_md,
            timestamp=timestamp,
            requested_modes=requested_modes,
            corpus_label=args.corpus_label,
            corpus_profile=metadata.get("corpus_profile"),
        )
        write_jsonl(snapshot_paths[0], records)
        write_markdown_report(
            snapshot_paths[1],
            records,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            base_url=args.base_url,
            requested_modes=requested_modes,
            metadata=metadata,
            corpus_label=args.corpus_label,
        )

    unsupported = sum(record["status"] == "unsupported" for record in records)
    errors = sum(record["status"] in {"error", "service_error"} for record in records)
    ok = sum(record["status"] == "ok" for record in records)
    print(
        f"Model matrix: {ok} ok, {unsupported} unsupported, {errors} errors "
        f"across {len(records)} query/mode pairs"
    )
    print(f"JSONL written to {args.out_jsonl}")
    print(f"Report written to {args.out_md}")
    if snapshot_paths is not None:
        print(f"Snapshot JSONL written to {snapshot_paths[0]}")
        print(f"Snapshot report written to {snapshot_paths[1]}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
