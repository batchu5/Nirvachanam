"""Bug detection agent — identifies potential bugs in PR diffs.

Uses Gemini (primary) or Groq (fallback) via QuotaAwareFallbackLLM.
Outputs structured list[Finding] validated with pydantic.

References: PRD §1b (model allocation), §2 (agent orchestration), §3 (Finding schema).
"""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import ValidationError

from src.llm.fallback import QuotaAwareFallbackLLM
from src.models.schemas import BugAgentOutput, DiffContext, Finding

logger = structlog.get_logger()

# Load prompt from versioned file
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "v1" / "bug_agent.md"


def _load_system_prompt() -> str:
    """Load the bug agent system prompt from the versioned prompt file."""
    return PROMPT_PATH.read_text(encoding="utf-8")


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


def _parse_response(raw_content: str) -> list[Finding]:
    """Parse and validate LLM response using Pydantic.

    Args:
        raw_content: JSON string from LLM response.

    Returns:
        List of validated Finding objects.

    Raises:
        ValidationError: If pydantic validation fails.
    """
    output = BugAgentOutput.model_validate_json(raw_content)
    return output.findings


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
        response_schema=BugAgentOutput,
        temperature=0.2,
        max_tokens=2000,
        timeout=timeout,
    )

    if response is None:
        logger.warning("bug_agent.all_providers_failed")
        return []

    # Parse and validate via Pydantic
    try:
        findings = _parse_response(response.content)
        logger.info(
            "bug_agent.completed",
            findings=len(findings),
            model=response.model,
            provider=response.provider,
            tokens=response.tokens_used,
        )
        return findings

    except ValidationError as e:
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
                    f"Your previous response failed validation. Error: {e}\n"
                    "Please try again with valid JSON matching the schema."
                ),
            },
        ]

        retry_response = await llm.invoke(
            messages=repair_messages,
            response_schema=BugAgentOutput,
            temperature=0.1,
            max_tokens=2000,
            timeout=timeout,
        )

        if retry_response is None:
            logger.warning("bug_agent.repair_retry_failed")
            return []

        try:
            findings = _parse_response(retry_response.content)
            logger.info(
                "bug_agent.completed_after_repair",
                findings=len(findings),
                model=retry_response.model,
            )
            return findings
        except Exception as e2:
            logger.error("bug_agent.repair_failed", error=str(e2))
            return []

