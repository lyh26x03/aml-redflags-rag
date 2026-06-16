"""Sanitize provider-facing errors before surfacing them in debug artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx


SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "key",
    "sig",
    "signature",
    "token",
}
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
BEARER_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"
)
SENSITIVE_PARAM_PATTERN = re.compile(
    r"(?i)\b("
    r"access_token|api_key|apikey|auth|authorization|key|sig|signature|token"
    r")=([^&\s]+)"
)


@dataclass(frozen=True)
class ProviderErrorDetails:
    provider: str
    model_name: str
    error_type: str
    message: str
    http_status: Optional[int] = None
    parse_success: Optional[bool] = None

    @property
    def fallback_reason(self) -> str:
        parts = [f"provider={self.provider or 'unknown'}"]
        if self.model_name:
            parts.append(f"model={self.model_name}")
        parts.append(f"error_type={self.error_type}")
        if self.http_status is not None:
            parts.append(f"http_status={self.http_status}")
        parts.append(f"message={self.message}")
        return " ".join(parts)


def sanitize_error_message(message: Any, max_length: int = 240) -> str | None:
    if message is None:
        return None
    text = str(message)
    if not text.strip():
        return None
    text = URL_PATTERN.sub(_sanitize_url_match, text)
    text = BEARER_PATTERN.sub("Bearer [redacted]", text)
    text = SENSITIVE_PARAM_PATTERN.sub(r"\1=[redacted]", text)
    text = " ".join(text.split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def build_provider_error_details(
    provider: str,
    model_name: str,
    *,
    error_type: str,
    message: Any,
    http_status: int | None = None,
    parse_success: bool | None = None,
) -> ProviderErrorDetails:
    sanitized = sanitize_error_message(message, max_length=160)
    return ProviderErrorDetails(
        provider=provider or "unknown",
        model_name=model_name,
        error_type=error_type,
        message=sanitized or "Provider error",
        http_status=http_status,
        parse_success=parse_success,
    )


def describe_provider_error(
    exc: Exception,
    *,
    provider: str,
    model_name: str,
) -> ProviderErrorDetails:
    if isinstance(exc, httpx.TimeoutException):
        return build_provider_error_details(
            provider,
            model_name,
            error_type="timeout",
            message="Provider request timed out",
        )

    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        return build_provider_error_details(
            provider,
            model_name,
            error_type="http_error",
            http_status=response.status_code if response is not None else None,
            message=_response_error_message(response),
        )

    if isinstance(exc, json.JSONDecodeError):
        return build_provider_error_details(
            provider,
            model_name,
            error_type="parse_error",
            parse_success=False,
            message="Provider response was not valid JSON",
        )

    if isinstance(exc, (KeyError, IndexError, TypeError)):
        return build_provider_error_details(
            provider,
            model_name,
            error_type="parse_error",
            parse_success=False,
            message="Provider response did not match the expected schema",
        )

    message = sanitize_error_message(exc, max_length=160) or exc.__class__.__name__
    if isinstance(exc, ValueError):
        lowered = message.lower()
        error_type = "value_error"
        if "api key is missing" in lowered:
            error_type = "missing_key"
        elif "unsupported llm provider" in lowered:
            error_type = "unsupported_provider"
        elif "model_name must be set" in lowered:
            error_type = "invalid_model_config"
        return build_provider_error_details(
            provider,
            model_name,
            error_type=error_type,
            message=message,
        )

    if isinstance(exc, httpx.RequestError):
        return build_provider_error_details(
            provider,
            model_name,
            error_type="request_error",
            message=message or "Provider request failed",
        )

    return build_provider_error_details(
        provider,
        model_name,
        error_type=_snake_case(exc.__class__.__name__),
        message=message,
    )


def _sanitize_url_match(match: re.Match[str]) -> str:
    return _sanitize_url(match.group(0))


def _sanitize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return SENSITIVE_PARAM_PATTERN.sub(r"\1=[redacted]", url)
    if not parts.query:
        return url

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    redacted_pairs = []
    for key, value in query_pairs:
        if key.lower() in SENSITIVE_QUERY_KEYS:
            redacted_pairs.append((key, "[redacted]"))
        else:
            redacted_pairs.append((key, value))

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(redacted_pairs, doseq=True),
            "",
        )
    )


def _extract_body_detail(body: Any) -> str | None:
    if isinstance(body, dict):
        parts: list[str] = []
        detail = body.get("detail")
        if isinstance(detail, list):
            for item in detail[:3]:
                if isinstance(item, dict):
                    loc = ".".join(str(part) for part in item.get("loc", []))
                    msg = item.get("msg")
                    if msg and loc:
                        parts.append(f"{loc}: {msg}")
                    elif msg:
                        parts.append(str(msg))
        elif isinstance(detail, str):
            parts.append(detail)

        error = body.get("error")
        if isinstance(error, dict):
            for key in ("message", "status", "code"):
                value = error.get(key)
                if value:
                    parts.append(str(value))
        elif isinstance(error, str):
            parts.append(error)

        for key in ("message", "status", "code"):
            value = body.get(key)
            if value:
                parts.append(str(value))

        if parts:
            return sanitize_error_message("; ".join(parts), max_length=160)

        return sanitize_error_message(
            json.dumps(body, ensure_ascii=False),
            max_length=160,
        )

    if isinstance(body, list):
        return sanitize_error_message(
            json.dumps(body[:3], ensure_ascii=False),
            max_length=160,
        )

    return sanitize_error_message(body, max_length=160)


def _response_error_message(response: httpx.Response | None) -> str:
    if response is None:
        return "Provider request failed"

    try:
        body = response.json()
    except ValueError:
        body = None

    detail = _extract_body_detail(body)
    if detail:
        return detail

    reason = getattr(response, "reason_phrase", None)
    if isinstance(reason, str) and reason.strip():
        sanitized = sanitize_error_message(reason, max_length=160)
        if sanitized:
            return sanitized

    text = sanitize_error_message(getattr(response, "text", None), max_length=160)
    if text:
        return text
    return "Provider request failed"


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
