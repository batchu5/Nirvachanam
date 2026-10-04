"""Tool call argument repair — fixes malformed LLM tool-call arguments.

When the LLM produces broken tool-call arguments (invalid JSON, wrong
field names, missing required fields), this layer attempts deterministic
repair before falling back to re-prompting the LLM.

This is separate from the validation retry in reviewer.py — it operates
at the individual tool-call argument level, not the whole response.

Repairs handled:
  1. Broken JSON (missing quotes, trailing commas, etc.)
  2. Missing required fields (fill with sensible defaults)
  3. Wrong field names (fuzzy match to closest valid name)
  4. Type coercion (string "42" → int 42, "true" → bool True)

References: new_architecture.md Phase 8 (Comment Repair Layer),
            Alibaba OCR comment_args_repair.go
"""

from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import Any

import structlog

logger = structlog.get_logger()


def repair_tool_call_args(
    raw_args: str,
    tool_schema: dict[str, Any],
) -> dict[str, Any] | None:
    """Attempt to repair malformed LLM tool-call arguments.

    Tries multiple repair strategies in order:
    1. Direct JSON parse (no repair needed)
    2. Fix common JSON syntax errors
    3. Extract JSON from markdown code blocks
    4. Fix field names via fuzzy matching
    5. Fill missing required fields with defaults
    6. Type coercion

    Args:
        raw_args: Raw argument string from the LLM tool call.
        tool_schema: The tool's parameter JSON Schema.

    Returns:
        Repaired argument dict, or None if repair failed.
    """
    # Step 1: Try direct parse
    parsed = _try_parse_json(raw_args)

    # Step 2: Fix common JSON issues and retry
    if parsed is None:
        fixed = _fix_json_syntax(raw_args)
        parsed = _try_parse_json(fixed)

    # Step 3: Extract from markdown code blocks
    if parsed is None:
        extracted = _extract_json_from_markdown(raw_args)
        if extracted:
            parsed = _try_parse_json(extracted)

    if parsed is None:
        logger.warning(
            "repair.json_parse_failed",
            raw_args_preview=raw_args[:200],
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning("repair.not_a_dict", type=type(parsed).__name__)
        return None

    # Step 4: Fix field names
    properties = tool_schema.get("properties", {})
    parsed = _fix_field_names(parsed, properties)

    # Step 5: Fill missing required fields
    required = tool_schema.get("required", [])
    parsed = _fill_missing_fields(parsed, properties, required)

    # Step 6: Type coercion
    parsed = _coerce_types(parsed, properties)

    # Final validation: check all required fields present
    missing = [r for r in required if r not in parsed]
    if missing:
        logger.warning("repair.missing_required_fields", missing=missing)
        return None

    logger.debug(
        "repair.success",
        fields=list(parsed.keys()),
    )
    return parsed


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _try_parse_json(s: str) -> dict[str, Any] | None:
    """Try to parse a string as JSON."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _fix_json_syntax(s: str) -> str:
    """Fix common JSON syntax errors.

    Handles:
    - Trailing commas before } or ]
    - Single quotes → double quotes
    - Missing quotes around keys
    - JavaScript-style comments
    """
    # Remove JS-style comments
    s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)

    # Trailing commas
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)

    # Single quotes → double quotes (naive but covers most cases)
    # Only replace if not inside a double-quoted string
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')

    # Unquoted keys: { key: "value" } → { "key": "value" }
    s = re.sub(
        r'(?<=\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
        r' "\1":',
        s,
    )

    return s


def _extract_json_from_markdown(s: str) -> str | None:
    """Extract JSON from markdown code blocks."""
    # Try ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', s, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try raw { ... } extraction
    brace_start = s.find('{')
    brace_end = s.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return s[brace_start:brace_end + 1]

    return None


def _fix_field_names(
    parsed: dict[str, Any],
    properties: dict[str, Any],
) -> dict[str, Any]:
    """Fix wrong field names by fuzzy matching to valid property names.

    If a key in the parsed dict doesn't match any property but is close
    to one (e.g., "filepath" → "file", "msg" → "message"), rename it.
    """
    valid_names = list(properties.keys())
    if not valid_names:
        return parsed

    repaired: dict[str, Any] = {}
    for key, value in parsed.items():
        if key in valid_names:
            repaired[key] = value
        else:
            # Try fuzzy match
            matches = get_close_matches(key, valid_names, n=1, cutoff=0.6)
            if matches:
                repaired_name = matches[0]
                if repaired_name not in repaired:  # Don't overwrite existing
                    logger.debug(
                        "repair.field_renamed",
                        original=key,
                        repaired=repaired_name,
                    )
                    repaired[repaired_name] = value
            else:
                # Keep unknown fields (might be optional extras)
                repaired[key] = value

    return repaired


# ---------------------------------------------------------------------------
# Missing field defaults
# ---------------------------------------------------------------------------

# Sensible defaults for common review tool fields
_FIELD_DEFAULTS: dict[str, Any] = {
    "severity": "warning",
    "category": "bug",
    "confidence": 0.7,
    "suggested_fix": None,
    "language": None,
    "start_line": None,
    "end_line": None,
    "file_pattern": "",
    "summary": "Review complete",
}


def _fill_missing_fields(
    parsed: dict[str, Any],
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    """Fill missing required fields with sensible defaults."""
    for field_name in required:
        if field_name not in parsed:
            if field_name in _FIELD_DEFAULTS:
                parsed[field_name] = _FIELD_DEFAULTS[field_name]
                logger.debug(
                    "repair.field_defaulted",
                    field=field_name,
                    default=_FIELD_DEFAULTS[field_name],
                )

    return parsed


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def _coerce_types(
    parsed: dict[str, Any],
    properties: dict[str, Any],
) -> dict[str, Any]:
    """Coerce field values to match the expected JSON Schema types.

    Handles: string "42" → int 42, "true" → bool True, etc.
    """
    for key, value in list(parsed.items()):
        if key not in properties:
            continue

        prop_schema = properties[key]
        expected_type = prop_schema.get("type")

        if expected_type == "integer" and isinstance(value, str):
            try:
                parsed[key] = int(value)
            except ValueError:
                pass

        elif expected_type == "number" and isinstance(value, str):
            try:
                parsed[key] = float(value)
            except ValueError:
                pass

        elif expected_type == "boolean" and isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                parsed[key] = True
            elif value.lower() in ("false", "0", "no"):
                parsed[key] = False

        elif expected_type == "number" and isinstance(value, int):
            parsed[key] = float(value)

        elif expected_type == "string" and not isinstance(value, str):
            parsed[key] = str(value)

    return parsed
