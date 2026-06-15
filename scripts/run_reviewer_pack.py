"""Run reviewer-oriented validation and write a concise Markdown report."""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "reports" / "reviewer_latest.md"
API_SMOKE_OUTPUT = REPO_ROOT / "eval" / "results" / "api_smoke_latest.jsonl"
CQC_OUTPUT = REPO_ROOT / "eval" / "results" / "cqc_latest.jsonl"
CQC_REPORT = REPO_ROOT / "eval" / "reports" / "cqc_latest.md"
INTERPRETATION_NOTE = (
    "This report is a local reviewer convenience artifact. It is not a "
    "model-quality benchmark and does not reproduce the historical "
    "private-corpus retrieval benchmark."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reviewer Demo Pack checks.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def run_command(command: list[str], timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "returncode": 124,
            "stdout": (error.stdout or "").strip(),
            "stderr": f"timed out after {timeout:g} seconds",
        }


def command_status(result: dict[str, Any]) -> str:
    return "PASS" if result["returncode"] == 0 else "FAIL"


def command_summary(result: dict[str, Any]) -> str:
    output = result["stdout"] or result["stderr"] or "no output"
    return output.splitlines()[-1]


def git_metadata(timeout: float) -> tuple[str, str]:
    branch = run_command(["git", "branch", "--show-current"], timeout)
    commit = run_command(["git", "rev-parse", "--short", "HEAD"], timeout)
    return (
        branch["stdout"] if branch["returncode"] == 0 else "unavailable",
        commit["stdout"] if commit["returncode"] == 0 else "unavailable",
    )


def probe_health(base_url: str, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as response:
            status_code = response.status
            raw_body = response.read().decode("utf-8")
    except HTTPError as error:
        status_code = error.code
        raw_body = error.read().decode("utf-8")
    except (URLError, TimeoutError, OSError) as error:
        return {
            "status": "WARN",
            "reachable": False,
            "http_status": None,
            "body": None,
            "message": (
                f"FastAPI service is not reachable: {error}. Start it with "
                "`.venv\\Scripts\\python.exe -m uvicorn api.main:app --reload`."
            ),
        }
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        body = raw_body
    healthy = (
        status_code == 200
        and isinstance(body, dict)
        and body.get("status") == "ok"
    )
    return {
        "status": "PASS" if healthy else "WARN",
        "reachable": True,
        "http_status": status_code,
        "body": body,
        "message": "Service is healthy." if healthy else "Service responded but is not healthy.",
    }


def overall_status(
    compile_result: dict[str, Any],
    pytest_result: dict[str, Any],
    skip_live: bool,
    health: dict[str, Any] | None,
    live_results: list[dict[str, Any]],
) -> str:
    if command_status(compile_result) == "FAIL" or command_status(pytest_result) == "FAIL":
        return "FAIL"
    if skip_live:
        return "PASS"
    if not health or not health["reachable"]:
        return "WARN"
    if health["status"] != "PASS" or any(command_status(result) == "FAIL" for result in live_results):
        return "WARN"
    return "PASS"


def _command_section(title: str, result: dict[str, Any]) -> list[str]:
    command = " ".join(result["command"])
    return [
        f"### {title}",
        "",
        f"- **Status:** {command_status(result)}",
        f"- **Command:** `{command}`",
        f"- **Summary:** {command_summary(result)}",
        "",
    ]


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Reviewer Demo Pack Report",
        "",
        f"- **Timestamp UTC:** {report['timestamp']}",
        f"- **Python:** {report['python_version']}",
        f"- **Branch:** `{report['branch']}`",
        f"- **Commit:** `{report['commit']}`",
        f"- **Overall status:** **{report['overall_status']}**",
        "",
        "## Static Validation",
        "",
    ]
    lines.extend(_command_section("Compileall", report["compileall"]))
    lines.extend(_command_section("Pytest", report["pytest"]))
    lines.extend(["## Live Service Validation", ""])

    if report["skip_live"]:
        lines.extend(
            [
                "- **Status:** SKIPPED",
                "- Live checks were explicitly skipped with `--skip-live`.",
                "",
            ]
        )
    else:
        health = report["health"]
        lines.extend(
            [
                f"- **Health status:** {health['status']}",
                f"- **Base URL:** `{report['base_url']}`",
                f"- **Details:** {health['message']}",
                "",
            ]
        )

    lines.extend(["## API Smoke Eval", ""])
    if report["api_smoke"]:
        lines.extend(_command_section("API smoke evaluation", report["api_smoke"]))
    else:
        lines.extend(["- Not run because live checks were skipped or the service was unreachable.", ""])

    lines.extend(["## CQC-RAG Lite Eval", ""])
    if report["cqc_eval"]:
        lines.extend(_command_section("CQC-RAG Lite evaluation", report["cqc_eval"]))
    else:
        lines.extend(["- Not run because live checks were skipped or the service was unreachable.", ""])

    lines.extend(["## Generated Artifacts", ""])
    for artifact in report["artifacts"]:
        state = "present" if artifact["present"] else "not present"
        lines.append(f"- `{artifact['path']}`: {state}")

    lines.extend(["", "## Interpretation Note", "", f"> {INTERPRETATION_NOTE}", ""])
    return "\n".join(lines)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")


def build_report(base_url: str, timeout: float, skip_live: bool) -> dict[str, Any]:
    python = sys.executable
    compile_result = run_command(
        [python, "-m", "compileall", "api", "rag_core", "indexing", "tests", "scripts"],
        timeout,
    )
    pytest_result = run_command([python, "-m", "pytest", "tests", "-q"], timeout)
    branch, commit = git_metadata(timeout)

    health = None
    api_smoke = None
    cqc_eval = None
    live_results: list[dict[str, Any]] = []
    if not skip_live:
        health = probe_health(base_url, timeout)
        if health["reachable"]:
            api_smoke = run_command(
                [
                    python,
                    "scripts/run_api_smoke_eval.py",
                    "--base-url",
                    base_url,
                    "--timeout",
                    str(timeout),
                ],
                timeout,
            )
            cqc_eval = run_command(
                [
                    python,
                    "scripts/run_cqc_eval.py",
                    "--base-url",
                    base_url,
                    "--timeout",
                    str(timeout),
                    "--report-md",
                    str(CQC_REPORT),
                ],
                timeout,
            )
            live_results = [api_smoke, cqc_eval]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "branch": branch,
        "commit": commit,
        "base_url": base_url,
        "skip_live": skip_live,
        "compileall": compile_result,
        "pytest": pytest_result,
        "health": health,
        "api_smoke": api_smoke,
        "cqc_eval": cqc_eval,
        "artifacts": [
            {"path": API_SMOKE_OUTPUT.relative_to(REPO_ROOT).as_posix(), "present": API_SMOKE_OUTPUT.exists()},
            {"path": CQC_OUTPUT.relative_to(REPO_ROOT).as_posix(), "present": CQC_OUTPUT.exists()},
            {"path": CQC_REPORT.relative_to(REPO_ROOT).as_posix(), "present": CQC_REPORT.exists()},
        ],
        "overall_status": overall_status(
            compile_result, pytest_result, skip_live, health, live_results
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(args.base_url, args.timeout, args.skip_live)
    write_report(args.output_md, report)
    print(f"Reviewer Demo Pack: {report['overall_status']}")
    print(f"Report written to {args.output_md}")
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
