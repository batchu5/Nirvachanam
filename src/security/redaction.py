"""Pre-LLM secret redaction — strips secrets from diff text before any agent sees it.

This is a critical security boundary: NO raw secret value should ever reach
a third-party LLM API. Instead, secrets are replaced with [REDACTED:type]
markers, and structured redaction records are passed to security_agent so
it can still report "secret detected at line X" without seeing the value.

References: PRD §15a.
"""

from __future__ import annotations

import re

import structlog

from src.models.schemas import RedactionRecord

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Secret detection patterns (§15a)
# ---------------------------------------------------------------------------
# Intentionally broad — better to over-redact than to leak a real secret
# to a third-party API. False positives just replace non-secret strings
# with [REDACTED], which agents can still reason about.

SECRET_PATTERNS: list[tuple[str, str]] = [
    # GitHub tokens (PAT, OAuth, etc.)
    (r"gh[ps]_[A-Za-z0-9_]{36,}", "GitHub Token"),
    # AWS Access Key IDs
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    # AWS Secret Access Keys
    (
        r"(?i)aws_secret_access_key\s*=\s*[\w/+=]{40}",
        "AWS Secret Key",
    ),
    # Private key PEM blocks
    (r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----", "Private Key PEM"),
    # Slack tokens
    (r"xox[baprs]-[A-Za-z0-9\-]{10,}", "Slack Token"),
    # Generic high-entropy secrets (passwords, tokens, API keys in assignments)
    (
        r"(?i)(?:api[_\-]?key|secret|token|password|auth)\s*[=:]\s*[\"'][A-Za-z0-9+/=_\-]{20,}[\"']",
        "Generic Secret",
    ),
    # High-entropy base64 in secret-like assignments
    (
        r"(?i)(?:secret|password|token)\s*[=:]\s*[\"'][A-Za-z0-9+/]{32,}={0,2}[\"']",
        "High-Entropy Secret",
    ),
]

# Pre-compile patterns for performance
_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern), label) for pattern, label in SECRET_PATTERNS
]


def redact_secrets(diff_text: str) -> tuple[str, list[RedactionRecord]]:
    """Scan diff text for secret patterns and replace with [REDACTED:type].

    Called BEFORE any diff content reaches any agent (including planner).
    The redaction records tell security_agent WHERE secrets were found
    (file, line, pattern type) without exposing the actual secret value.

    Args:
        diff_text: Raw diff / patch text.

    Returns:
        Tuple of:
        - Redacted diff text (secrets replaced with [REDACTED:type] markers)
        - List of RedactionRecord objects describing what was redacted
    """
    if not diff_text:
        return diff_text, []

    redactions: list[RedactionRecord] = []
    redacted = diff_text

    for compiled_pattern, label in _COMPILED_PATTERNS:
        for match in compiled_pattern.finditer(diff_text):
            # Replace the matched secret with a redaction marker
            redacted = redacted.replace(
                match.group(0),
                f"[REDACTED:{label}]",
            )
            # Approximate line number from character offset
            line_num = diff_text[: match.start()].count("\n") + 1
            redactions.append(
                RedactionRecord(
                    line=line_num,
                    type=label,
                    length=len(match.group(0)),
                )
            )

    if redactions:
        logger.warning(
            "security.secrets_redacted",
            count=len(redactions),
            types=[r.type for r in redactions],
        )

    return redacted, redactions
