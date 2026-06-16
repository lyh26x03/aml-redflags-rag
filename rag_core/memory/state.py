"""Bounded structured conversation state.

``ConversationMemory`` preserves the *useful* AML conversation state across
turns without ever storing an unlimited raw transcript:

- active scenario summary           (``active_scenario_summary``)
- active red flags (deduplicated)   (``active_flags``)
- previous citations (bounded)      (``active_citations``)
- previous retrieved chunk IDs      (``retrieved_chunk_ids``)
- prior assessment                  (``last_assessment``)
- referenceable prior answer        (``last_answer_summary``)
- unresolved clarification needs    (``unresolved_questions``)
- recent bounded turn summaries     (``recent_turns``)

Every list is bounded and every free-text field is truncated. Mutations go
through ``record_*`` helpers so the bounds are always enforced in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# --- bounds (exported for tests and docs) ---
MAX_RECENT_TURNS = 8
MAX_ACTIVE_CITATIONS = 10
MAX_ACTIVE_FLAGS = 12
MAX_RETRIEVED_CHUNK_IDS = 30
MAX_CONTEXT_TERMS = 20
MAX_UNRESOLVED_QUESTIONS = 10

CITATION_EXCERPT_CHARS = 200
QUERY_SUMMARY_CHARS = 300
ANSWER_SUMMARY_CHARS = 400
SCENARIO_SUMMARY_CHARS = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: Any, limit: int) -> str:
    """Collapse whitespace and hard-truncate to ``limit`` characters."""
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "…"


@dataclass
class MemoryCitation:
    """A bounded reference to a previously cited evidence chunk."""

    chunk_id: str
    source: str
    excerpt: str

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "MemoryCitation":
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            source=str(payload.get("source", "Unknown")),
            excerpt=_truncate(payload.get("excerpt", ""), CITATION_EXCERPT_CHARS),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "excerpt": self.excerpt,
        }


@dataclass
class TurnSummary:
    """A concise, bounded summary of a single conversation turn.

    This is deliberately *not* the raw transcript — only the fields needed to
    reason about the conversation later are retained.
    """

    turn_index: int
    intent_route: str
    user_query: str
    assessment: Optional[str]
    answer_summary: str
    flag_codes: List[str]
    citation_chunk_ids: List[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "intent_route": self.intent_route,
            "user_query": self.user_query,
            "assessment": self.assessment,
            "answer_summary": self.answer_summary,
            "flag_codes": list(self.flag_codes),
            "citation_chunk_ids": list(self.citation_chunk_ids),
            "created_at": self.created_at,
        }


@dataclass
class ConversationMemory:
    """Bounded structured state for one ``session_id``."""

    session_id: str
    turn_count: int = 0
    recent_turns: List[TurnSummary] = field(default_factory=list)
    active_scenario_summary: str = ""
    active_entities_or_context_terms: List[str] = field(default_factory=list)
    active_flags: List[Dict[str, str]] = field(default_factory=list)
    active_citations: List[MemoryCitation] = field(default_factory=list)
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    last_assessment: Optional[str] = None
    last_answer_summary: str = ""
    unresolved_questions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    # --- queries ---

    @property
    def has_content(self) -> bool:
        """True once at least one substantive turn has been recorded."""
        return self.turn_count > 0

    @property
    def active_flag_codes(self) -> List[str]:
        return [str(flag.get("code", "")) for flag in self.active_flags]

    # --- recording (each enforces the bounds) ---

    def record_retrieval_turn(
        self,
        *,
        intent_route: str,
        user_query: str,
        answer: str,
        assessment: Optional[str],
        flags: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        retrieved_chunk_ids: List[str],
        context_terms: List[str],
    ) -> None:
        """Record an evidence-retrieval turn — updates the active scenario."""
        scenario = _truncate(user_query, SCENARIO_SUMMARY_CHARS)
        if scenario:
            self.active_scenario_summary = scenario
        self.last_answer_summary = _truncate(answer, ANSWER_SUMMARY_CHARS)
        self.last_assessment = assessment

        self._merge_flags(flags)
        self._merge_citations(citations)
        self._merge_chunk_ids(retrieved_chunk_ids)
        self._merge_context_terms(context_terms)

        flag_codes = [str(flag.get("code", "")) for flag in flags if flag.get("code")]
        citation_ids = [
            str(c.get("chunk_id", "")) for c in citations if c.get("chunk_id")
        ]
        self._append_turn(
            intent_route=intent_route,
            user_query=user_query,
            assessment=assessment,
            answer_summary=self.last_answer_summary,
            flag_codes=flag_codes,
            citation_chunk_ids=citation_ids,
        )

    def record_clarify_turn(self, *, user_query: str, clarification: str) -> None:
        """Record an under-specified turn — stores the unresolved need.

        Recorded under the ``ask_clarifying_question`` route: the system asks the
        user for the missing detail rather than fabricating an assessment.
        """
        self._add_unresolved(user_query)
        self._append_turn(
            intent_route="ask_clarifying_question",
            user_query=user_query,
            assessment="unlikely",
            answer_summary=_truncate(clarification, ANSWER_SUMMARY_CHARS),
            flag_codes=[],
            citation_chunk_ids=[],
        )

    def record_history_turn(
        self,
        *,
        user_query: str,
        answer: str,
        assessment: Optional[str],
    ) -> None:
        """Record an answer-from-history turn — does not change the scenario."""
        self._append_turn(
            intent_route="answer_from_history",
            user_query=user_query,
            assessment=assessment,
            answer_summary=_truncate(answer, ANSWER_SUMMARY_CHARS),
            flag_codes=list(self.active_flag_codes),
            citation_chunk_ids=[c.chunk_id for c in self.active_citations],
        )

    def resolve_clarifications(self) -> None:
        """Clear outstanding clarification needs once a real scenario lands."""
        self.unresolved_questions = []

    # --- bounded mutators ---

    def _append_turn(
        self,
        *,
        intent_route: str,
        user_query: str,
        assessment: Optional[str],
        answer_summary: str,
        flag_codes: List[str],
        citation_chunk_ids: List[str],
    ) -> None:
        self.turn_count += 1
        self.recent_turns.append(
            TurnSummary(
                turn_index=self.turn_count,
                intent_route=intent_route,
                user_query=_truncate(user_query, QUERY_SUMMARY_CHARS),
                assessment=assessment,
                answer_summary=answer_summary,
                flag_codes=list(dict.fromkeys(flag_codes)),
                citation_chunk_ids=list(dict.fromkeys(citation_chunk_ids)),
                created_at=_now_iso(),
            )
        )
        # keep only the most recent N turns (bounded raw history)
        if len(self.recent_turns) > MAX_RECENT_TURNS:
            self.recent_turns = self.recent_turns[-MAX_RECENT_TURNS:]
        self._touch()

    def _merge_flags(self, flags: List[Dict[str, Any]]) -> None:
        by_code: Dict[str, Dict[str, str]] = {
            str(flag.get("code", "")): flag for flag in self.active_flags
        }
        for flag in flags:
            code = str(flag.get("code", "")).strip()
            if not code:
                continue
            by_code[code] = {
                "code": code,
                "name": str(flag.get("name", "")),
                "name_zh": str(flag.get("name_zh") or ""),
            }
        merged = list(by_code.values())
        self.active_flags = merged[-MAX_ACTIVE_FLAGS:]

    def _merge_citations(self, citations: List[Dict[str, Any]]) -> None:
        by_chunk: "Dict[str, MemoryCitation]" = {
            citation.chunk_id: citation for citation in self.active_citations
        }
        for payload in citations:
            citation = MemoryCitation.from_payload(payload)
            if not citation.chunk_id:
                continue
            # re-insert so the most recently cited chunks stay at the end
            by_chunk.pop(citation.chunk_id, None)
            by_chunk[citation.chunk_id] = citation
        merged = list(by_chunk.values())
        self.active_citations = merged[-MAX_ACTIVE_CITATIONS:]

    def _merge_chunk_ids(self, chunk_ids: List[str]) -> None:
        ordered = list(self.retrieved_chunk_ids)
        for chunk_id in chunk_ids:
            chunk_id = str(chunk_id)
            if not chunk_id:
                continue
            if chunk_id in ordered:
                ordered.remove(chunk_id)
            ordered.append(chunk_id)
        self.retrieved_chunk_ids = ordered[-MAX_RETRIEVED_CHUNK_IDS:]

    def _merge_context_terms(self, terms: List[str]) -> None:
        ordered = list(self.active_entities_or_context_terms)
        for term in terms:
            term = str(term).strip()
            if not term or term in ordered:
                continue
            ordered.append(term)
        self.active_entities_or_context_terms = ordered[-MAX_CONTEXT_TERMS:]

    def _add_unresolved(self, question: str) -> None:
        summary = _truncate(question, QUERY_SUMMARY_CHARS)
        if not summary or summary in self.unresolved_questions:
            return
        self.unresolved_questions.append(summary)
        self.unresolved_questions = self.unresolved_questions[-MAX_UNRESOLVED_QUESTIONS:]

    def _touch(self) -> None:
        self.updated_at = _now_iso()

    # --- serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "active_scenario_summary": self.active_scenario_summary,
            "active_entities_or_context_terms": list(
                self.active_entities_or_context_terms
            ),
            "active_flags": [dict(flag) for flag in self.active_flags],
            "active_citations": [c.to_dict() for c in self.active_citations],
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "last_assessment": self.last_assessment,
            "last_answer_summary": self.last_answer_summary,
            "unresolved_questions": list(self.unresolved_questions),
            "recent_turns": [turn.to_dict() for turn in self.recent_turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
