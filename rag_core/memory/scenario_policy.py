"""Deterministic scenario-state update policy for multi-turn AML memory.

This module isolates the *decision* of how the active AML case state should
evolve from turn to turn. The original code collapsed three different kinds of
information into a single slot (``active_scenario_summary``):

- the **case backbone** — the stable summary of the AML case under analysis;
- the **per-turn delta** — the new condition a follow-up turn introduces;
- the **raw turn query** — what the user typed this turn.

Overwriting the backbone with the raw follow-up query on every
``retrieve_with_memory`` turn caused scenario drift from turn 3 onward (the
retrieval query degraded into a string of disconnected follow-up fragments).
See ``docs/active_scenario_summary_overwrite_test_report.md``.

The policy here decides exactly one action on the backbone per turn:

``SEED``     no case yet → adopt this turn's query as the backbone.
``PRESERVE`` keep the backbone, record this turn only as a bounded delta
             (the default for follow-ups and refinements of the same case).
``REPLACE``  a genuinely new standalone case → replace the backbone and clear
             the case-scoped deltas so stale context cannot leak forward.
``REPAIR``   the backbone has drifted into a degenerate fragment → rebuild it
             (defensive; the new policy prevents drift, the detector guards it).
``NOOP``     nothing actionable (e.g. an empty query).

It is **rule-based and deterministic** (no live LLM), so routing and memory
remain testable and audit-friendly. The new-case judgement is factored behind
a pluggable ``new_case_scorer`` seam so a learned scorer could later replace the
deterministic default without touching any caller — the deterministic rule is
always the fallback (constraint: never make the policy depend on a live LLM
without a deterministic fallback).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Set

# --- bounds (exported via the memory package for tests and docs) -------------
MAX_CASE_DELTAS = 6
DELTA_SUMMARY_CHARS = 200

# Drift-detection thresholds.
MIN_BACKBONE_CHARS = 12
DRIFT_TERM_RETENTION_MIN = 0.34  # backbone must retain >= this share of case terms
NEW_CASE_TOPIC_OVERLAP_MAX = 0.20  # topic Jaccard <= this (with no additive cue)
NEW_CASE_MIN_WORDS = 6

# Must match ``rag_core.intent_router.ROUTE_RETRIEVE_WITH_MEMORY``. Kept as a
# local literal so the memory package stays decoupled from the router module.
FOLLOWUP_ROUTE = "retrieve_with_memory"

# --- scenario update actions -------------------------------------------------
ACTION_SEED = "seed"
ACTION_PRESERVE = "preserve"
ACTION_REPLACE = "replace"
ACTION_REPAIR = "repair"
ACTION_NOOP = "noop"

ALL_SCENARIO_ACTIONS = (
    ACTION_SEED,
    ACTION_PRESERVE,
    ACTION_REPLACE,
    ACTION_REPAIR,
    ACTION_NOOP,
)

# Cues that a plain ``retrieve`` turn is *adding to the same case* rather than
# opening a new one ("the customer also…", "另外…"). These force PRESERVE.
_ADDITIVE_CUE = re.compile(
    "|".join(
        [
            r"\balso\b",
            r"\bin addition\b",
            r"\badditionally\b",
            r"\bas well\b",
            r"\bfurthermore\b",
            r"\bmoreover\b",
            r"\bon top of (that|this)\b",
            r"另外",
            r"此外",
            r"而且",
            r"並且",
            r"還有",
            r"再加上",
            r"同時",
            r"也(有|涉及|出現|存在|包括)",
        ]
    ),
    re.IGNORECASE,
)

# Phrasings that, standing alone, are clearly a *follow-up question* rather than
# a case summary. Used by the drift detector to catch a backbone that collapsed
# into a follow-up fragment.
_FRAGMENT_CUE = re.compile(
    "|".join(
        [
            r"^\s*what about\b",
            r"^\s*how about\b",
            r"^\s*and what about\b",
            r"\bis (this|that) related\b",
            r"\bdoes (this|that) relate\b",
            r"^那(跟|和|與|這|是不是|算不算|會不會|有沒有)",
            r"^這(跟|和|與)",
            r"有關(係)?嗎\s*$",
            r"相關嗎\s*$",
        ]
    ),
    re.IGNORECASE,
)

# Leading follow-up connectors stripped when distilling a delta so the stored
# refinement reads as the *new condition* ("profile mismatch") rather than the
# connector ("what about profile mismatch?"). Substantive terms are never
# removed, so retrieval signal is preserved.
_LEADING_CONNECTOR = re.compile(
    r"^\s*(?:"
    r"and what about|what about|how about|"
    r"is (?:this|that) related to|does (?:that|this) relate to|related to|"
    r"那跟|那和|那與|這跟|這和|這與|那是不是|那會不會|那算不算|那有沒有"
    r")\s*",
    re.IGNORECASE,
)


def distill_delta(query: str) -> str:
    """Reduce a follow-up query to its new condition for the delta store.

    Conservative: strips a known leading connector and trailing punctuation,
    and falls back to the original text if stripping would empty it.
    """
    text = " ".join(str(query or "").split())
    if not text:
        return ""
    cleaned = _LEADING_CONNECTOR.sub("", text).strip()
    cleaned = cleaned.strip(" ?？.。!！，,、").strip()
    return cleaned or text


@dataclass(frozen=True)
class ScenarioDecision:
    """The chosen action on the case backbone, with an audit reason."""

    action: str
    reason: str

    @property
    def composes_case_context(self) -> bool:
        """Whether this turn should retrieve with the prior case context.

        True exactly when the turn continues the established case (PRESERVE).
        A SEED (first case) or REPLACE (new case) retrieves on its own query.
        """
        return self.action == ACTION_PRESERVE


def _as_set(values: Iterable[str]) -> Set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _rule_based_new_case_score(
    *,
    current_query: str,
    new_topics: Set[str],
    existing_topics: Set[str],
) -> float:
    """Deterministic new-standalone-case score in ``[0, 1]``.

    Returns 1.0 only when the turn is a self-contained AML scenario whose
    topics are essentially disjoint from the established case — i.e. a new case,
    not a refinement. Otherwise 0.0 (treat as a refinement, never silently lose
    the prior case).
    """
    word_count = len(current_query.split())
    substantive = word_count >= NEW_CASE_MIN_WORDS or len(new_topics) >= 2
    if not substantive:
        return 0.0
    if not new_topics:
        # No AML topic of its own → cannot be a confident new case.
        return 0.0
    if _jaccard(new_topics, existing_topics) <= NEW_CASE_TOPIC_OVERLAP_MAX:
        return 1.0
    return 0.0


NewCaseScorer = Callable[..., float]


def decide_scenario_update(
    *,
    current_query: str,
    route: str,
    has_backbone: bool,
    new_topics: Iterable[str],
    existing_topics: Iterable[str],
    new_case_scorer: Optional[NewCaseScorer] = None,
) -> ScenarioDecision:
    """Decide how the active case backbone should change on this turn.

    Pure and deterministic. ``new_topics``/``existing_topics`` are detected AML
    topic keys (from :class:`rag_core.gate.TopicDetector`); ``route`` is the
    fine-grained intent route. ``new_case_scorer`` overrides the deterministic
    new-case rule (the rule remains the fallback when it returns ``None``-like).
    """
    text = " ".join(str(current_query or "").split())
    if not text:
        return ScenarioDecision(ACTION_NOOP, "empty_query")

    if not has_backbone:
        return ScenarioDecision(ACTION_SEED, "first_case")

    # A memory follow-up always builds on the established case: never overwrite.
    if route == FOLLOWUP_ROUTE:
        return ScenarioDecision(ACTION_PRESERVE, "memory_followup")

    # Plain retrieve with an existing case.
    if _ADDITIVE_CUE.search(text):
        return ScenarioDecision(ACTION_PRESERVE, "additive_refinement")

    scorer = new_case_scorer or _rule_based_new_case_score
    score = scorer(
        current_query=text,
        new_topics=_as_set(new_topics),
        existing_topics=_as_set(existing_topics),
    )
    if score is not None and score >= 0.5:
        return ScenarioDecision(ACTION_REPLACE, "new_standalone_case")
    return ScenarioDecision(ACTION_PRESERVE, "related_retrieve")


# --- drift detection (Option D: memory failure detector) ---------------------


def _terms(text: str) -> Set[str]:
    """Lowercased significant tokens (len >= 3) for term-overlap diagnostics."""
    return {tok for tok in re.split(r"[^0-9a-z一-鿿]+", (text or "").lower()) if len(tok) >= 3}


@dataclass(frozen=True)
class ScenarioDriftReport:
    """Diagnostic snapshot of the case backbone's health.

    ``term_retention`` is the share of the reference case's significant terms
    still present in the backbone (1.0 when there is no reference to compare).
    """

    drift: bool
    severity: str  # "none" | "low" | "high"
    reasons: Sequence[str]
    term_retention: float

    def to_dict(self) -> dict:
        return {
            "drift": self.drift,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "term_retention": round(self.term_retention, 3),
        }


def detect_scenario_drift(
    *,
    backbone: str,
    reference: str = "",
) -> ScenarioDriftReport:
    """Detect whether the case backbone has degraded into a follow-up fragment.

    ``reference`` is the original case text (the seed turn). The check is
    deterministic token-overlap — a lightweight stand-in for the
    representation-space probe described in the test report — and never raises.
    """
    backbone = " ".join(str(backbone or "").split())
    reasons = []
    high = False

    if not backbone:
        return ScenarioDriftReport(True, "high", ("empty_backbone",), 0.0)

    if _FRAGMENT_CUE.search(backbone):
        reasons.append("backbone_is_followup_fragment")
        high = True

    if len(backbone) < MIN_BACKBONE_CHARS:
        reasons.append("backbone_too_short")

    reference = " ".join(str(reference or "").split())
    ref_terms = _terms(reference)
    if ref_terms:
        retention = len(ref_terms & _terms(backbone)) / len(ref_terms)
    else:
        retention = 1.0
    if ref_terms and retention < DRIFT_TERM_RETENTION_MIN:
        reasons.append("lost_original_case_terms")
        if retention < NEW_CASE_TOPIC_OVERLAP_MAX:
            high = True

    if not reasons:
        return ScenarioDriftReport(False, "none", (), retention)
    return ScenarioDriftReport(True, "high" if high else "low", tuple(reasons), retention)
