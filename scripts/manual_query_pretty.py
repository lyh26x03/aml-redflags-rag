"""Pretty-print a /query response for manual reviewer inspection.

Usage:
    .venv\\Scripts\\python.exe scripts\\manual_query_pretty.py [options]

Options:
    --base-url        API base URL (default: http://localhost:8000)
    --query           Query string
    --retrieval-mode  hybrid | dense | bm25
    --llm-mode        mock | gemini | gemma | groq | ollama
    --top-k           Number of chunks to retrieve (int)
    --session-id      Session ID for conversation memory
    --use-memory      Enable structured conversation memory (flag)
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


DEFAULTS = {
    "base_url": "http://localhost:8000",
    "query": "Funds show rapid movement through a virtual asset exchange.",
    "retrieval_mode": "hybrid",
    "llm_mode": "mock",
    "top_k": 5,
}


def _col(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _hdr(title: str) -> None:
    print(_col("1;36", f"\n{'─' * 60}"))
    print(_col("1;36", f"  {title}"))
    print(_col("1;36", f"{'─' * 60}"))


def _field(label: str, value: object) -> None:
    label_str = _col("33", f"{label}:")
    print(f"  {label_str} {value}")


def print_response(data: dict) -> None:
    _hdr("ASSESSMENT")
    _field("assessment", _col("1;32" if data.get("assessment") == "possible" else "1;33", data.get("assessment", "—")))

    flags = data.get("identified_flags") or []
    _hdr(f"IDENTIFIED FLAGS  ({len(flags)})")
    if flags:
        for f in flags:
            print(f"  • [{_col('35', f.get('code', '?'))}] {f.get('name', '?')}")
    else:
        print("  (none)")

    citations = data.get("citations") or []
    _hdr(f"CITATIONS  ({len(citations)})")
    if citations:
        for c in citations:
            print(f"  • chunk_id={_col('35', c.get('chunk_id', '?'))}  source={c.get('source', '?')}")
    else:
        print("  (none)")

    refusal = data.get("refusal") or {}
    _hdr("REFUSAL")
    _field("refused", refusal.get("refused"))
    _field("reason", refusal.get("reason") or "—")

    debug = data.get("debug") or {}
    _hdr("DEBUG")
    for key in (
        "retrieval_mode",
        "llm_mode",
        "fallback_used",
        "fallback_reason",
        "intent_route",
        "route_family",
        "memory_used",
        "retrieved_chunk_ids",
    ):
        val = debug.get(key)
        if val is None:
            val = "—"
        _field(key, val)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULTS["base_url"])
    parser.add_argument("--query", default=DEFAULTS["query"])
    parser.add_argument("--retrieval-mode", default=DEFAULTS["retrieval_mode"])
    parser.add_argument("--llm-mode", default=DEFAULTS["llm_mode"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--use-memory", action="store_true", default=False)
    args = parser.parse_args()

    payload: dict = {
        "query": args.query,
        "retrieval_mode": args.retrieval_mode,
        "llm_mode": args.llm_mode,
        "include_debug": True,
        "top_k": args.top_k,
        "use_memory": args.use_memory,
    }
    if args.session_id:
        payload["session_id"] = args.session_id

    url = args.base_url.rstrip("/") + "/query"
    body = json.dumps(payload).encode()

    print(_col("1", f"\nPOST {url}"))
    print(_col("90", f"     {json.dumps(payload, separators=(',', ':'))}"))

    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(_col("1;31", f"\nHTTP {exc.code}: {exc.reason}"), file=sys.stderr)
        try:
            print(exc.read().decode(), file=sys.stderr)
        except Exception:
            pass
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(_col("1;31", f"\nConnection error: {exc.reason}"), file=sys.stderr)
        print("Is the API running?  Try:  uvicorn api.main:app --reload", file=sys.stderr)
        sys.exit(1)

    print_response(data)


if __name__ == "__main__":
    main()
