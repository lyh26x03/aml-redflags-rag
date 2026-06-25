# CLAUDE.md

## Documentation sync rule

Whenever the user requests code modifications, automatically check and update the
relevant project notes and documentation upon completion to ensure they never
become outdated.

Specifically: after any change to `rag_core/`, `api/`, `scripts/`, or `tests/`,
review and update the affected sections in `docs/`, `MIGRATION_INVENTORY.md`,
and `README.md` to reflect the new implementation state.

## Read-only files

`.py`, `.env`, and core configuration files are read-only unless the user
explicitly grants permission to modify them.

## Key documentation map

| When you change... | Also review... |
|---|---|
| `rag_core/memory/` or `rag_core/intent_router.py` | `docs/conversation_memory.md`, `README.md` memory section |
| `rag_core/generation.py` | `README.md` generation modes, `docs/demo_spec.md` addendum |
| `rag_core/retrieval.py` | `docs/evaluation_notes.md`, `README.md` retrieval section |
| `rag_core/gate.py` | `docs/demo_spec.md` RAG behavior section |
| `api/main.py` | `docs/demo_spec.md` API design, `docs/demo_evidence_pack.md` |
| `scripts/` | `README.md` eval sections, `docs/reviewer_guide.md` |
| Any new feature | `MIGRATION_INVENTORY.md` deferred/experimental table, `README.md` status matrix |
