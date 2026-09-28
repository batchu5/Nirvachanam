"""Bug detection agent — identifies potential bugs in PR diffs.

Uses Gemini (primary) or Groq (fallback) via QuotaAwareFallbackLLM.
Outputs structured list[Finding] validated with pydantic.

References: PRD §1b (model allocation), §2 (agent orchestration), §3 (Finding schema).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from pydantic import ValidationError

from src.llm.fallback import QuotaAwareFallbackLLM
from src.models.enums import Category, Severity
from src.models.schemas import DiffContext, Finding

logger = structlog.get_logger()

# Load prompt from versioned file
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "v1" / "bug_agent.md"

# JSON Schema for structured output — matches Finding model
FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                    "category": {"type": "string", "enum": ["bug"]},
                    "message": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "confidence": {"type": "number"},
                    "agent": {"type": "string", "enum": ["bug_agent"]},
                    "language": {"type": "string"},
                },
                "required": ["file", "line", "severity", "category", "message", "agent"],
            },
        }
    },
    "required": ["findings"],
}


def _load_system_prompt() -> str:
    """Load the bug agent system prompt from the versioned prompt file."""
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    # Fallback inline prompt if file not found
    logger.warning("bug_agent.prompt_file_missing", path=str(PROMPT_PATH))
    return (
        "You are a code review bug detection agent. Analyze the diff and return "
        "a JSON object with a 'findings' array of bugs found. Each finding must have: "
        "file, line, severity (info/warning/critical), category (bug), message, agent (bug_agent)."
    )


def _build_diff_content(diff_context: DiffContext) -> str:
    """Build the diff content string for the agent prompt.

    Includes file patches wrapped in XML delimiters for injection defense.
    """
    parts: list[str] = []
    for file_ctx in diff_context.files:
        if file_ctx.patch:
            parts.append(f"### File: {file_ctx.path}")
            if file_ctx.language:
                parts.append(f"Language: {file_ctx.language}")
            parts.append(file_ctx.patch)
            parts.append("")  # Blank separator

    return "\n".join(parts)


def _parse_findings(raw_content: str) -> list[Finding]:
    """Parse LLM response into validated Finding objects.

    Args:
        raw_content: JSON string from LLM response.

    Returns:
        List of validated Finding objects.

    Raises:
        ValueError: If JSON parsing fails.
        ValidationError: If pydantic validation fails.
    """
    data = json.loads(raw_content)
    findings_data = data.get("findings", [])

    findings: list[Finding] = []
    for item in findings_data:
        # Ensure required defaults
        item.setdefault("category", "bug")
        item.setdefault("agent", "bug_agent")
        item.setdefault("confidence", 0.5)

        try:
            finding = Finding(**item)
            findings.append(finding)
        except ValidationError as e:
            logger.warning(
                "bug_agent.finding_validation_error",
                error=str(e),
                raw=item,
            )
            # Skip invalid findings rather than crash

    return findings


async def run_bug_agent(
    diff_context: DiffContext,
    llm: QuotaAwareFallbackLLM,
    timeout: int = 30,
) -> list[Finding]:
    """Run the bug detection agent on a diff.

    Args:
        diff_context: Parsed and redacted diff context.
        llm: QuotaAwareFallbackLLM instance for LLM calls.
        timeout: Hard timeout for the LLM call.

    Returns:
        List of validated Finding objects. Returns empty list on failure
        (agent skipped, not crashed).
    """
    system_prompt = _load_system_prompt()
    diff_content = _build_diff_content(diff_context)

    if not diff_content.strip():
        logger.info("bug_agent.empty_diff")
        return []

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<user_diff>\n{diff_content}\n</user_diff>"},
    ]

    # First attempt
    response = await llm.invoke(
        messages=messages,
        response_schema=FINDING_SCHEMA,
        temperature=0.2,
        max_tokens=2000,
        timeout=timeout,
    )

    if response is None:
        logger.warning("bug_agent.all_providers_failed")
        return []

    # Parse and validate
    try:
        findings = _parse_findings(response.content)
        logger.info(
            "bug_agent.completed",
            findings=len(findings),
            model=response.model,
            provider=response.provider,
            tokens=response.tokens_used,
        )
        return findings

    except (json.JSONDecodeError, ValueError) as e:
        # Repair retry: inject error into prompt and try again
        logger.warning("bug_agent.parse_error_retrying", error=str(e))

        repair_messages = messages + [
            {
                "role": "assistant",
                "content": response.content,
            },
            {
                "role": "user",
                "content": (
                    f"Your previous response was not valid JSON. Error: {e}\n"
                    "Please try again with valid JSON matching the schema."
                ),
            },
        ]

        retry_response = await llm.invoke(
            messages=repair_messages,
            response_schema=FINDING_SCHEMA,
            temperature=0.1,
            max_tokens=2000,
            timeout=timeout,
        )

        if retry_response is None:
            logger.warning("bug_agent.repair_retry_failed")
            return []

        try:
            findings = _parse_findings(retry_response.content)
            logger.info(
                "bug_agent.completed_after_repair",
                findings=len(findings),
                model=retry_response.model,
            )
            return findings
        except Exception as e2:
            logger.error("bug_agent.repair_failed", error=str(e2))
            return []

    except ValidationError as e:
        logger.error("bug_agent.validation_error", error=str(e))
        return []
