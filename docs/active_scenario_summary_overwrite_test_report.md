# active_scenario_summary Overwrite Test Report

- Date: 2026-06-26 13:37:07 +08:00 (original failure snapshot)
- Branch: `memory-router-state-fix`
- Scope: Multi-turn structured memory behavior only
- Core implementation files modified: none at snapshot time — the fix is
  documented in [Resolution (2026-06-26)](#resolution-2026-06-26) below

## Objective

Verify that `active_scenario_summary` continues to represent the active AML
case after consecutive `retrieve_with_memory` follow-ups, rather than being
replaced by the most recent short question.

## Test Design

| ID | Layer | Scenario | Expected invariant |
|---|---|---|---|
| AS-01 | Memory state | Record a substantive retrieval turn, then a short `retrieve_with_memory` follow-up | The active summary still contains the original case terms. |
| AS-02 | Pipeline | Run a case plus two follow-ups through a capturing retriever | The third retrieval query still contains the original case terms. |
| AS-03 | API / BM25 | Run the same three turns through the FastAPI service and inspect the memory endpoint | The session snapshot still contains the original case terms. |

Test scenario:

1. `Funds show rapid movement through a virtual asset exchange.`
2. `What about profile mismatch?`
3. `What about cross-border transfers?`

The original case terms checked by the tests are `rapid movement` and
`virtual asset`.

## Execution

Targeted regression tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_active_scenario_summary_overwrite.py -vv
```

Result: `3 failed`

Complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

Result: `3 failed, 121 passed, 2 warnings`

The 3 failures are the new AS-01 through AS-03 regression tests. The existing
121 tests pass when the new regression file is excluded.

## Observed Evidence

### AS-01: Memory State

After turn 2, `active_scenario_summary` is:

```text
What about profile mismatch?
```

It no longer contains `rapid movement` or `virtual asset`.

### AS-02: Pipeline Retrieval Query

The second retrieval query is correct:

```text
Funds show rapid movement through a virtual asset exchange.
What about profile mismatch?
```

The third retrieval query has drifted:

```text
What about profile mismatch?
What about cross-border transfers?
```

It no longer includes the original case terms.

### AS-03: API Memory Snapshot

After turn 3, `GET /sessions/summary-overwrite-api/memory` reports:

```text
active_scenario_summary = What about cross-border transfers?
```

The turn routes are still correct: turns 2 and 3 both use
`retrieve_with_memory`.

## Interpretation

The tests confirm a memory state update defect, not a routing or retrieval
algorithm defect:

1. The router correctly selects `retrieve_with_memory` for follow-ups.
2. The second turn correctly retrieves with the original case plus the new
   question.
3. The state after that turn replaces the case summary with the follow-up.
4. The third retrieval therefore receives only consecutive follow-up phrases.

`active_flags`, citations, and retrieved chunk IDs can still accumulate, which
can make memory debug data appear healthy while the next retrieval query has
lost the case context.

## Test Status

The regression suite is intentionally red until the active scenario state
update policy preserves or meaningfully merges the original case summary for
memory-assisted follow-ups.

No changes were made to:

- `rag_core/pipeline.py`
- `rag_core/memory/state.py`
- `rag_core/intent_router.py`
- `rag_core/retrieval.py`
- `api/main.py`

## Resolution (2026-06-26)

The defect was fixed by redesigning the scenario state as a **case backbone +
bounded deltas** governed by a deterministic policy, rather than overwriting
one slot with the latest query.

### Root cause

The failure was not in BM25, dense retrieval, or LLM generation. The structured
memory had an unstable update policy: `record_retrieval_turn` set
`active_scenario_summary = current_user_query` on every retrieval turn, so a
short `retrieve_with_memory` follow-up overwrote the case summary. Turn 2 still
worked (it composed the prior summary), but from turn 3 the retrieval query
degraded into a chain of follow-up fragments.

### Fix

- New `rag_core/memory/scenario_policy.py`: a pure, deterministic policy
  (`decide_scenario_update`) that chooses `seed` / `preserve` / `replace` /
  `repair` / `noop`, plus delta distillation and a drift detector
  (`detect_scenario_drift`). The new-case judgement sits behind a pluggable
  `new_case_scorer` seam with the rule-based scorer as the fallback.
- `rag_core/memory/state.py`: split the case into `active_scenario_summary`
  (stable backbone) and `active_case_deltas` (bounded follow-up refinements).
  A follow-up appends a delta; only a new standalone case replaces the backbone.
  Added `compose_retrieval_query`, `scenario_health`, and `repair_scenario`.
- `rag_core/pipeline.py`: `retrieve_with_memory` now composes
  `backbone + deltas + current_query`; new `debug.scenario_update_action` /
  `debug.case_delta_count` audit fields.
- `rag_core/intent_router.py`: whole-case questions ("combined red flags",
  "綜合來看…") route to `retrieve_with_memory` when memory exists, so they
  retrieve *with* the accumulated case context.
- `rag_core/schemas.py`: additive debug + snapshot fields (`active_case_deltas`,
  `scenario_health`, etc.). No existing field renamed or removed.

### Test status — now green

The original AS-01..AS-03 regressions pass, and the coverage was extended well
beyond the third turn:

- `tests/test_active_scenario_summary_overwrite.py` — original 3 regressions (pass)
- `tests/test_memory_scenario_policy.py` — 24 tests: policy units, distillation,
  drift detector, state-level multi-turn, router whole-case routing, a
  four-turn pipeline drift test, and the six requirement scenarios (Tests 1–6)
- `tests/test_memory_hardening.py` — 8 adversarial tests: drift corruption +
  recovery, replace-then-follow-up, 50-turn bounded growth, delta dedup,
  Chinese follow-ups, and the conservative no-drop safety bias

Full suite: `156 passed` (was `3 failed, 121 passed`).
