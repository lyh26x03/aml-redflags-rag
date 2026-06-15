"""Tests for the reviewer-oriented validation pack."""

from pathlib import Path

from scripts import run_reviewer_pack


REPO_ROOT = Path(__file__).resolve().parents[1]


def _command(returncode=0, stdout="completed", stderr=""):
    return {
        "command": ["python", "-m", "example"],
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _report():
    return {
        "timestamp": "2026-06-15T07:00:00+00:00",
        "python_version": "3.12.0",
        "branch": "reviewer-demo-pack",
        "commit": "abc1234",
        "base_url": "http://localhost:8000",
        "skip_live": False,
        "compileall": _command(stdout="compile passed"),
        "pytest": _command(stdout="17 passed"),
        "health": {
            "status": "PASS",
            "reachable": True,
            "message": "Service is healthy.",
        },
        "api_smoke": _command(stdout="API smoke eval: 8 / 8 passed"),
        "cqc_eval": _command(stdout="CQC-RAG lite: 5 / 5 groups passed"),
        "artifacts": [
            {"path": "eval/results/api_smoke_latest.jsonl", "present": True},
            {"path": "eval/results/cqc_latest.jsonl", "present": True},
            {"path": "eval/reports/cqc_latest.md", "present": True},
        ],
        "overall_status": "PASS",
    }


def test_render_markdown_report_with_synthetic_results():
    rendered = run_reviewer_pack.render_markdown_report(_report())

    assert rendered.startswith("# Reviewer Demo Pack Report\n")
    assert "**Overall status:** **PASS**" in rendered
    assert "## Static Validation" in rendered
    assert "## Live Service Validation" in rendered
    assert "## API Smoke Eval" in rendered
    assert "## CQC-RAG Lite Eval" in rendered
    assert run_reviewer_pack.INTERPRETATION_NOTE in rendered


def test_command_result_summarization():
    assert run_reviewer_pack.command_status(_command()) == "PASS"
    assert run_reviewer_pack.command_status(_command(returncode=1)) == "FAIL"
    assert run_reviewer_pack.command_summary(_command(stdout="first\nlast")) == "last"
    assert run_reviewer_pack.command_summary(_command(stdout="", stderr="failed")) == "failed"


def test_unreachable_service_is_warn_not_fail_when_static_checks_pass():
    health = {"status": "WARN", "reachable": False}

    status = run_reviewer_pack.overall_status(
        _command(), _command(), False, health, []
    )

    assert status == "WARN"


def test_write_report_creates_requested_parent_path(tmp_path):
    output = tmp_path / "nested" / "reviewer_latest.md"

    run_reviewer_pack.write_report(output, _report())

    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("# Reviewer Demo Pack Report")


def test_generated_latest_reports_are_ignored():
    ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "eval/reports/*_latest.md" in ignore_text
    assert "eval/reports/reviewer_latest.md" in ignore_text
    assert "eval/results/*_latest.jsonl" in ignore_text
