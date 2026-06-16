"""Structured, bounded, in-process conversation memory for the AML demo.

This package adds multi-turn *structured* conversation state to the
single-turn RAG service. It is intentionally **not** a production memory
store: state lives in process, is bounded, and is never persisted.

Public surface:
- ``ConversationMemory``      — the bounded per-session state object
- ``MemoryCitation`` / ``TurnSummary`` — bounded sub-records
- ``ConversationMemoryStore`` — thread-safe in-process session registry
- bound constants (``MAX_RECENT_TURNS`` etc.) for tests/docs
"""

from rag_core.memory.state import (
    ANSWER_SUMMARY_CHARS,
    CITATION_EXCERPT_CHARS,
    MAX_ACTIVE_CITATIONS,
    MAX_ACTIVE_FLAGS,
    MAX_CONTEXT_TERMS,
    MAX_RECENT_TURNS,
    MAX_RETRIEVED_CHUNK_IDS,
    MAX_UNRESOLVED_QUESTIONS,
    QUERY_SUMMARY_CHARS,
    SCENARIO_SUMMARY_CHARS,
    ConversationMemory,
    MemoryCitation,
    TurnSummary,
)
from rag_core.memory.store import ConversationMemoryStore

__all__ = [
    "ConversationMemory",
    "MemoryCitation",
    "TurnSummary",
    "ConversationMemoryStore",
    "MAX_RECENT_TURNS",
    "MAX_ACTIVE_CITATIONS",
    "MAX_ACTIVE_FLAGS",
    "MAX_RETRIEVED_CHUNK_IDS",
    "MAX_CONTEXT_TERMS",
    "MAX_UNRESOLVED_QUESTIONS",
    "CITATION_EXCERPT_CHARS",
    "QUERY_SUMMARY_CHARS",
    "ANSWER_SUMMARY_CHARS",
    "SCENARIO_SUMMARY_CHARS",
]
