"""Evidence-bound generation for the AML red-flag demo.

Mock mode is deterministic and never calls the network.

Live Groq / Gemini paths:
  - Key-gated: if GROQ_API_KEY or GEMINI_API_KEY is absent the call is skipped
    and mock output is returned with fallback_used=True in the debug block.
  - Not execution-verified: live providers were not called during development
    (no API keys available in that environment).
  - Output constraint: ``_normalize_live_result`` intersects the live LLM's
    identified flags with those that mock (keyword-based) evidence supports.
    This prevents hallucinated citations but can also drop valid flags the LLM
    identifies through reasoning that the keyword detector misses.
  - Any provider error, timeout, or malformed JSON falls back to mock with an
    explicit fallback_reason in the debug block.
"""

import json
from typing import Any, Dict, List, Optional, Set

import httpx

from rag_core.gate import TopicDetector


RF_CATALOG: Dict[str, Dict[str, str]] = {
    "RF-01": {"name": "Structuring", "name_zh": "門檻拆分"},
    "RF-02": {"name": "Rapid Movement", "name_zh": "快速流轉"},
    "RF-03": {"name": "Unusual Cash Activity", "name_zh": "現金異常"},
    "RF-04": {"name": "Third-Party Control", "name_zh": "第三人代辦"},
    "RF-05": {"name": "High-Risk Cross-Border Activity", "name_zh": "跨境高風險"},
    "RF-06": {"name": "Profile Mismatch", "name_zh": "與身分不符"},
    "RF-07": {"name": "Virtual Asset Anonymity", "name_zh": "虛擬資產匿名"},
    "RF-08": {"name": "Opaque Ownership", "name_zh": "公司不透明"},
}

TOPIC_TO_FLAGS: Dict[str, Set[str]] = {
    "cash_structuring": {"RF-01"},
    "rapid_movement": {"RF-02"},
    "third_party": {"RF-04"},
    "cross_border": {"RF-05"},
    "identity_mismatch": {"RF-06"},
    "virtual_assets": {"RF-07"},
    "shell_company": {"RF-08"},
}

SYSTEM_PROMPT = """\
You are an AML (anti-money-laundering) red-flag analysis assistant.

# Red-flag catalog
RF-01 Structuring — deposits/withdrawals split to avoid reporting thresholds
RF-02 Rapid Movement — funds transit quickly through accounts with little retention
RF-03 Unusual Cash Activity — cash volumes inconsistent with the customer profile
RF-04 Third-Party Control — account operated or opened by an unrelated third party
RF-05 High-Risk Cross-Border Activity — transfers to/from high-risk jurisdictions
RF-06 Profile Mismatch — transaction pattern inconsistent with stated occupation/business
RF-07 Virtual Asset Anonymity — mixing services, non-custodial wallets, or privacy coins
RF-08 Opaque Ownership — shell companies or structures where beneficial owner is unclear

# Task
Analyse the provided scenario using ONLY the retrieved evidence chunks below.
Identify which red flags (if any) the evidence supports.

# Output contract
Return a single JSON object with exactly these keys:
  "answer"           — 1–3 sentence plain-language summary (English or Chinese)
  "assessment"       — one of: "possible", "unlikely", "refuse"
  "identified_flags" — array of objects: {code, name, reason}
  "citations"        — array of objects: {chunk_id, source, excerpt}

# Hard rules
- Assessment MUST be "possible", "unlikely", or "refuse". Never "confirmed".
- Cite only chunk_id values that appear in the supplied evidence.
- If evidence is insufficient, return "unlikely" with an empty flags array.
- If the scenario is outside AML scope, return "refuse" and explain in "answer".
- Never invent facts, sources, or citations not present in the evidence.\
"""


def build_user_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Build the evidence prompt, adapted from the notebook function."""
    evidence = []
    for index, chunk in enumerate(chunks, start=1):
        evidence.append(
            "\n".join(
                [
                    f"Evidence {index}",
                    f"Source: {chunk.get('source', 'Unknown')}, page {chunk.get('page', 0)}",
                    f"Category: {chunk.get('doc_category', 'unknown')}",
                    f"Content: {chunk.get('text', '')}",
                ]
            )
        )
    return (
        f"Scenario:\n{query}\n\nRetrieved evidence:\n"
        + "\n\n".join(evidence)
        + "\n\nReturn JSON only."
    )


def _query_candidate_flags(query: str) -> Set[str]:
    topics = TopicDetector().detect_topics(query)
    return {
        flag
        for topic in topics
        for flag in TOPIC_TO_FLAGS.get(topic, set())
    }


def _chunk_flags(chunk: Dict[str, Any]) -> Set[str]:
    return {
        flag
        for flag in chunk.get("related_flags", [])
        if flag in RF_CATALOG
    }


def _citation(chunk: Dict[str, Any]) -> Dict[str, str]:
    text = " ".join(str(chunk.get("text", "")).split())
    return {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "source": str(chunk.get("source", "Unknown")),
        "excerpt": text[:200],
    }


def mock_generate(
    query: str,
    chunks: List[Dict[str, Any]],
    gate_allowed: bool = True,
    gate_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a deterministic response from query and chunk evidence."""
    if not gate_allowed:
        reason = gate_reason or "The request is outside the available knowledge scope."
        return {
            "answer": reason,
            "assessment": "refuse",
            "identified_flags": [],
            "citations": [],
            "refusal": {"refused": True, "reason": reason},
        }

    query_flags = _query_candidate_flags(query)
    evidence_flags = {
        flag for chunk in chunks for flag in _chunk_flags(chunk)
    }
    candidate_flags = sorted(query_flags & evidence_flags)

    if not chunks or not candidate_flags:
        return {
            "answer": (
                "The retrieved evidence is insufficient to identify a supported "
                "AML red flag for this scenario."
            ),
            "assessment": "unlikely",
            "identified_flags": [],
            "citations": [],
            "refusal": {"refused": False, "reason": None},
        }

    supporting_chunks = [
        chunk for chunk in chunks if _chunk_flags(chunk) & set(candidate_flags)
    ]
    identified_flags = []
    for flag in candidate_flags:
        catalog = RF_CATALOG[flag]
        source = next(
            chunk.get("source", "retrieved evidence")
            for chunk in supporting_chunks
            if flag in _chunk_flags(chunk)
        )
        identified_flags.append(
            {
                "code": flag,
                "name": catalog["name"],
                "name_zh": catalog["name_zh"],
                "reason": f"The scenario and retrieved evidence from {source} support this indicator.",
            }
        )

    names = ", ".join(f"{flag} {RF_CATALOG[flag]['name']}" for flag in candidate_flags)
    return {
        "answer": (
            f"The available evidence supports a possible AML red-flag assessment: {names}. "
            "This demo result is evidence-oriented and does not confirm wrongdoing."
        ),
        "assessment": "possible",
        "identified_flags": identified_flags,
        "citations": [_citation(chunk) for chunk in supporting_chunks],
        "refusal": {"refused": False, "reason": None},
    }


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    api_key: str,
    timeout: float,
) -> Dict[str, Any]:
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    api_key: str,
    timeout: float,
) -> Dict[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )
    response = httpx.post(
        url,
        params={"key": api_key},
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    llm_config: Dict[str, str],
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Call a supported live provider using plain HTTP."""
    provider = llm_config.get("provider", "")
    model_name = llm_config.get("llm_model_name", "")
    api_key = llm_config.get("api_key", "")
    if not api_key:
        raise ValueError(f"{provider or 'LLM'} API key is missing")
    if provider == "groq":
        return _call_groq(system_prompt, user_prompt, model_name, api_key, timeout)
    if provider == "gemini":
        return _call_gemini(system_prompt, user_prompt, model_name, api_key, timeout)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _normalize_live_result(
    live_result: Dict[str, Any],
    query: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep live output inside the demo response and evidence contract."""
    supported = mock_generate(query=query, chunks=chunks)
    allowed_codes = {flag["code"] for flag in supported["identified_flags"]}
    live_codes = {
        item.get("code")
        for item in live_result.get("identified_flags", [])
        if isinstance(item, dict)
    }
    codes = sorted(allowed_codes & live_codes)
    flags = [
        flag for flag in supported["identified_flags"] if flag["code"] in codes
    ]
    assessment = live_result.get("assessment")
    if assessment not in {"possible", "unlikely", "refuse"}:
        assessment = "possible" if flags else "unlikely"
    if assessment == "possible" and not flags:
        assessment = "unlikely"
    return {
        "answer": str(
            live_result.get("answer")
            or live_result.get("scenario_summary")
            or supported["answer"]
        ),
        "assessment": assessment,
        "identified_flags": flags,
        "citations": supported["citations"] if flags else [],
        "refusal": {
            "refused": assessment == "refuse",
            "reason": live_result.get("reason") if assessment == "refuse" else None,
        },
    }


def generate(
    query: str,
    chunks: List[Dict[str, Any]],
    llm_mode: str = "mock",
    model_name: str = "mock-local",
    gemini_api_key: str = "",
    groq_api_key: str = "",
    gate_allowed: bool = True,
    gate_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a response and attach internal fallback metadata."""
    mock = mock_generate(query, chunks, gate_allowed, gate_reason)
    if llm_mode == "mock" or not gate_allowed:
        return {
            **mock,
            "_generation_debug": {
                "requested_llm_mode": llm_mode,
                "effective_llm_mode": "mock",
                "fallback_used": False,
                "fallback_reason": None,
            },
        }

    api_key = gemini_api_key if llm_mode == "gemini" else groq_api_key
    provider_model = model_name
    if not provider_model or provider_model == "mock-local":
        provider_model = (
            "gemini-2.0-flash" if llm_mode == "gemini"
            else "llama-3.3-70b-versatile"
        )

    try:
        live = call_llm(
            SYSTEM_PROMPT,
            build_user_prompt(query, chunks),
            {
                "provider": llm_mode,
                "llm_model_name": provider_model,
                "api_key": api_key,
            },
        )
        normalized = _normalize_live_result(live, query, chunks)
        return {
            **normalized,
            "_generation_debug": {
                "requested_llm_mode": llm_mode,
                "effective_llm_mode": llm_mode,
                "fallback_used": False,
                "fallback_reason": None,
            },
        }
    except Exception as exc:
        return {
            **mock,
            "_generation_debug": {
                "requested_llm_mode": llm_mode,
                "effective_llm_mode": "mock",
                "fallback_used": True,
                "fallback_reason": str(exc),
            },
        }
