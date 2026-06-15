# Failure Diagnostics Lite

## Why This Exists

Failure Diagnostics Lite turns existing API smoke and CQC-RAG Lite result
traces into reviewer-facing debugging categories. It helps a reviewer move
from a failed check to the observable field or consistency metric worth
inspecting next.

This is an evaluation and observability layer. It does not change the RAG
algorithm, API behavior, or default API-key-free mock mode.

## What It Reads

- `eval/results/api_smoke_latest.jsonl`
- `eval/results/cqc_latest.jsonl`

Missing or malformed inputs are reported as diagnostics rather than causing
the report process to crash. The script never calls the FastAPI service.

## Diagnostic Categories

The deterministic taxonomy includes API smoke failures, unavailable inputs,
retrieval fallback, low citation overlap, retrieved-chunk instability,
identified-flag instability, assessment instability, refusal instability,
expected-behavior mismatch, insufficient evidence, and out-of-scope refusal.

Retrieval fallback and expected refusal can be normal behavior. Their presence
is a signal to inspect, not automatic proof of incorrect behavior.

## How To Run

```powershell
.venv\Scripts\python.exe scripts\run_failure_diagnostics.py
```

Generated local artifacts:

```text
eval/reports/failure_diagnostics_latest.md
eval/results/failure_diagnostics_latest.jsonl
```

Use `--strict` to return exit code 1 when any HIGH severity issue is found:

```powershell
.venv\Scripts\python.exe scripts\run_failure_diagnostics.py --strict
```

Without `--strict`, the script returns 0 when parsing and report writing
complete, even when diagnostics are present.

## How To Interpret Results

Start with HIGH issues, then inspect the named record, group, metric, and
evidence fields. A failed CQC group is a debugging entry point, not a final
quality score. Missing inputs usually mean the corresponding evaluator has not
been run locally.

This report is a local diagnostic artifact. It is not a model-quality
benchmark, does not reproduce the historical private-corpus retrieval
benchmark, and does not establish final AML correctness.

## What It Does Not Claim

- It is not a new RAG algorithm or a full CQC-RAG implementation.
- It does not modify retrieval, generation, gate, or pipeline behavior.
- It does not infer hidden model behavior beyond observable result fields.
- It does not claim production AML readiness or model quality.

## Future Extensions

Future work could add trend comparisons across explicitly versioned runs,
links from categories to richer trace views, and diagnostics for future public
evaluation corpora without changing the service pipeline.
