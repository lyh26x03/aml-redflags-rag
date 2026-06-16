"""Deterministic intent routing for multi-turn AML analysis.

This is a *rule-based* router by design: routing must be testable and must not
depend on a live LLM (constraint: avoid live-LLM dependency for routing). It
maps a user query — plus the current gate decision and whether memory is
available — onto one of five fine-grained routes.

Reviewer-facing model: every route collapses onto one of **three** high-level
outcomes (see :func:`route_family`):

- ``retrieve``              — evidence retrieval happened (with or without memory)
- ``refuse``                — request was out of scope
- ``no_retrieval_response`` — answered deterministically from conversation state
                              (history recall or asking the user to clarify)

Fine-grained routes (kept for debug + deterministic tests):
- ``retrieve``                — normal evidence retrieval (single-turn default)
- ``refuse``                  — clearly out-of-scope request
- ``ask_clarifying_question`` — vague/under-specified first-turn input: ask the
                                user for the missing detail and store the
                                unresolved need
- ``answer_from_history``     — user asks to recall/explain the previous
                                answer/flags/citations; answered from memory
- ``retrieve_with_memory``    — follow-up needing prior scenario context + fresh
                                retrieval

``ask_clarifying_question`` and ``answer_from_history`` are deliberately
distinct: the old notebook conflated both under a single "clarification" label.
Here, asking the user for more detail (``ask_clarifying_question``) is separate
from explaining a prior answer (``answer_from_history``).

The extra memory routes (ask_clarifying_question / answer_from_history /
retrieve_with_memory, and the router's own non-AML out-of-scope refusal) only
fire when memory is enabled, so single-turn ``/query`` behavior is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rag_core.gate import TopicDetector

ROUTE_RETRIEVE = "retrieve"
ROUTE_REFUSE = "refuse"
ROUTE_ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
ROUTE_ANSWER_FROM_HISTORY = "answer_from_history"
ROUTE_RETRIEVE_WITH_MEMORY = "retrieve_with_memory"

ALL_ROUTES = (
    ROUTE_RETRIEVE,
    ROUTE_REFUSE,
    ROUTE_ASK_CLARIFYING_QUESTION,
    ROUTE_ANSWER_FROM_HISTORY,
    ROUTE_RETRIEVE_WITH_MEMORY,
)

# --- high-level route families (reviewer-facing "three outcomes") -----------
# Docs and demos lead with these three outcomes; ``intent_route`` keeps the
# finer label for debugging and deterministic tests. The mapping is total and
# deterministic — no live-LLM involvement.
FAMILY_RETRIEVE = "retrieve"
FAMILY_REFUSE = "refuse"
FAMILY_NO_RETRIEVAL = "no_retrieval_response"

ALL_ROUTE_FAMILIES = (FAMILY_RETRIEVE, FAMILY_REFUSE, FAMILY_NO_RETRIEVAL)

_ROUTE_TO_FAMILY = {
    ROUTE_RETRIEVE: FAMILY_RETRIEVE,
    ROUTE_RETRIEVE_WITH_MEMORY: FAMILY_RETRIEVE,
    ROUTE_REFUSE: FAMILY_REFUSE,
    ROUTE_ANSWER_FROM_HISTORY: FAMILY_NO_RETRIEVAL,
    ROUTE_ASK_CLARIFYING_QUESTION: FAMILY_NO_RETRIEVAL,
}


def route_family(route: Optional[str]) -> Optional[str]:
    """Collapse a fine-grained route onto one of three reviewer-facing outcomes.

    Returns ``None`` only when ``route`` is ``None`` (e.g. debug omitted).
    Unknown routes default to the retrieve family, which is the safe
    single-turn default.
    """
    if route is None:
        return None
    return _ROUTE_TO_FAMILY.get(route, FAMILY_RETRIEVE)


def _compile(patterns) -> "re.Pattern[str]":
    return re.compile("|".join(patterns), re.IGNORECASE)


# References to a previous answer / previous flags / previous citations.
# e.g. "剛剛那個風險可以再說明嗎？", "剛剛引用的是哪些來源？", "those flags".
HISTORY_REFERENCE = _compile(
    [
        r"剛剛",
        r"剛才",
        r"上一題",
        r"上一個",
        r"上一則",
        r"上一輪",
        r"前一題",
        r"前面(提到|說|講|的)",
        r"之前(提到|說|講|的|問)",
        r"你(剛|剛剛|前面|提到|說過|講過)",
        r"再(說明|解釋|講)(一次|一下)?",
        r"重複(一次|一下)?",
        r"那(個|些)?(風險|紅旗|旗標|結果|判斷|分析|來源|引用|證據)",
        r"哪些(來源|引用|證據|紅旗|旗標|風險)",
        r"引用(的)?(來源|是哪|哪些|什麼)",
        r"\bprevious(ly)?\b",
        r"\bearlier\b",
        r"\bthat (evidence|flag|assessment|answer|result)\b",
        r"\bthose (flags?|citations?|sources?|red flags?)\b",
        r"\bwhich (sources?|citations?|flags?)\b",
        r"\bthe (citations?|sources?|evidence)\b",
        r"\byou (just )?(mentioned|said|identified|listed)\b",
        r"\b(last|prior) (answer|assessment|result)\b",
    ]
)

# Phrasings that explicitly ask about the cited sources / evidence, used to
# flag ``referenced_previous_evidence`` on an answer-from-history turn.
EVIDENCE_REFERENCE = _compile(
    [
        r"來源",
        r"引用",
        r"證據",
        r"出處",
        r"\bsources?\b",
        r"\bcitations?\b",
        r"\bevidence\b",
    ]
)

# Follow-up connectors: a new but related question that builds on the prior
# scenario. e.g. "那跟客戶職業不符有關嗎？", "what about cross-border?".
FOLLOWUP_CONNECTOR = _compile(
    [
        r"那(跟|和|與|這|是不是|算不算|會不會|有沒有)",
        r"這(跟|和|與|個跟|樣跟)",
        r"跟.{0,12}(有關|相關|有沒有關)",
        r"有關(係)?嗎",
        r"相關嗎",
        r"還有(呢|沒有)",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\b(is|does) that relate(d)?\b",
        r"\brelated to\b",
        r"\band what\b",
    ]
)

# Under-specified / vague phrasings that cannot be responsibly assessed.
# e.g. "這樣有沒有問題？".
VAGUE_PATTERN = _compile(
    [
        r"^這樣",
        r"這樣(有沒有問題|有問題嗎|可以嗎|對嗎|行嗎|算嗎|ok嗎|好嗎)",
        r"有沒有問題",
        r"有問題嗎",
        r"這個(可以嗎|行嗎|對嗎|有問題嗎)",
        r"^(可以嗎|行嗎|對嗎|怎麼樣|如何)",
        r"幫我看看",
        r"\bis this (ok|okay|fine|a problem|alright)\b",
        r"\bany (problem|issue)s?\b",
        r"\bwhat do you think\b",
        r"\bis it (ok|okay|fine)\b",
    ]
)

# Clearly non-AML chit-chat / unrelated tasks. Only treated as out-of-scope
# when no AML topic is detected, so legitimate AML queries are never caught.
OUT_OF_SCOPE = _compile(
    [
        r"推薦",
        r"晚餐",
        r"午餐",
        r"早餐",
        r"餐廳",
        r"美食",
        r"食譜",
        r"料理",
        r"煮",
        r"天氣",
        r"笑話",
        r"寫(一首|首)?詩",
        r"唱歌",
        r"講(個)?故事",
        r"旅遊",
        r"訂(機票|飯店|餐廳|位)",
        r"\brecommend\b",
        r"\b(dinner|lunch|breakfast|meal|restaurant|recipe)\b",
        r"\bwhat (should|can) i (eat|cook)\b",
        r"\bweather\b",
        r"\b(tell|write) me a (joke|poem|story)\b",
        r"\btranslate\b",
        r"\bbook a (flight|hotel|table)\b",
    ]
)


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    referenced_history: bool = False
    referenced_evidence: bool = False


class IntentRouter:
    """Maps a query to a :class:`RouteDecision` with no network calls."""

    def __init__(self, topic_detector: Optional[TopicDetector] = None):
        self.topic_detector = topic_detector or TopicDetector()

    def route(
        self,
        query: str,
        *,
        gate_allowed: bool,
        memory_enabled: bool,
        has_memory: bool,
    ) -> RouteDecision:
        text = query or ""

        # The AML scope gate (sanctions / TBML / tax) always wins.
        if not gate_allowed:
            return RouteDecision(ROUTE_REFUSE, "gate_out_of_scope")

        # Without memory the service stays single-turn: only retrieve/refuse.
        if not memory_enabled:
            return RouteDecision(ROUTE_RETRIEVE, "single_turn_default")

        detected_topics = self.topic_detector.detect_topics(text)
        is_history = bool(HISTORY_REFERENCE.search(text))
        is_followup = bool(FOLLOWUP_CONNECTOR.search(text))
        is_vague = bool(VAGUE_PATTERN.search(text))
        is_out_of_scope = bool(OUT_OF_SCOPE.search(text)) and not detected_topics

        # 1. Clearly non-AML request → refuse (and do not pollute AML memory).
        if is_out_of_scope and not is_history and not is_followup:
            return RouteDecision(ROUTE_REFUSE, "router_out_of_scope")

        # 2. Explicit recall of the previous answer / flags / citations.
        if is_history:
            return RouteDecision(
                ROUTE_ANSWER_FROM_HISTORY,
                "history_reference",
                referenced_history=True,
                referenced_evidence=bool(EVIDENCE_REFERENCE.search(text)),
            )

        # 3. Follow-up that needs the prior scenario context + fresh retrieval.
        if is_followup and has_memory:
            return RouteDecision(
                ROUTE_RETRIEVE_WITH_MEMORY,
                "followup_with_memory",
                referenced_history=True,
            )

        # 4. Under-specified query with no AML topic → ask a clarifying question.
        if is_vague and not detected_topics:
            return RouteDecision(ROUTE_ASK_CLARIFYING_QUESTION, "underspecified")

        # 5. Default: normal evidence retrieval.
        return RouteDecision(ROUTE_RETRIEVE, "default_retrieve")
