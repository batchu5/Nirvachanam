"""Unified reviewer agent — single-pass comprehensive code review.

Replaces the separate bug_agent, security_agent, style_agent, and test_agent
with a SINGLE LLM call that covers all categories. This reduces token usage
by ~80% since the diff is only sent once instead of 4 times.

Supports two review depths via prompt selection:
  - FAST: lightweight prompt, ~1,000 output tokens (critical issues only)
  - STANDARD/DEEP: full prompt, ~2,000 output tokens (all categories)

References: Token optimization strategy §1 (single-pass mega-prompt).
"""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import ValidationError

from src.agents.triage import ReviewTier
from src.llm.fallback import QuotaAwareFallbackLLM
from src.models.agent_io import ReviewerOutput, AgentFinding
from src.models.schemas import DiffContext, Finding
from src.models.enums import Category

logger = structlog.get_logger()

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts" / "v1"
PROMPT_STANDARD = PROMPT_DIR / "reviewer.md"
PROMPT_FAST = PROMPT_DIR / "reviewer_fast.md"


def _load_system_prompt(tier: ReviewTier) -> str:
    """Load the appropriate system prompt based on review tier."""
    if tier == ReviewTier.FAST:
        path = PROMPT_FAST
    else:
        path = PROMPT_STANDARD

    if path.exists():
        return path.read_text(encoding="utf-8")

    # Fallback inline prompt
    return (
        "You are an expert code reviewer. Analyze the diff for bugs, "
        "security issues, style problems, and test coverage gaps. "
        "Return JSON with a 'findings' array."
    )


def _build_diff_content(diff_context: DiffContext) -> str:
    """Build the diff content string for the reviewer prompt.

    Includes all file patches in a single block, plus redaction records.
    The diff is sent exactly ONCE — the whole point of the single-pass approach.
    """
    parts: list[str] = []

    for file_ctx in diff_context.files:
        if file_ctx.is_deleted:
            continue
        if file_ctx.patch:
            parts.append(f"### File: {file_ctx.path}")
            if file_ctx.language:
                parts.append(f"Language: {file_ctx.language}")
            parts.append(file_ctx.patch)
            parts.append("")

    # Add redaction records as metadata
    if diff_context.redaction_records:
        parts.append("\n### Pre-Redacted Secrets Detected")
        parts.append(
            "The following secrets were automatically redacted before analysis:"
        )
        for record in diff_context.redaction_records:
            parts.append(
                f"- Line {record.line}: {record.type} ({record.length} chars)"
            )

    return "\n".join(parts)


# Category string → Category enum mapping
_CATEGORY_MAP = {
    "bug": Category.BUG,
    "security": Category.SECURITY,
    "style": Category.STYLE,
    "test": Category.TEST,
}


def _to_findings(agent_findings: list[AgentFinding]) -> list[Finding]:
    """Convert AgentFinding list to Finding list.

    Maps category strings to the correct Category enum and sets the
    agent name to 'reviewer' for all findings.
    """
    findings: list[Finding] = []
    for af in agent_findings:
        category = _CATEGORY_MAP.get(
            str(af.category).lower(), Category.BUG
        )
        findings.append(
            Finding(
                file=af.file,
                line=af.line,
                end_line=af.end_line,
                severity=af.severity,
                category=category,
                message=af.message,
                suggested_fix=af.suggested_fix,
                confidence=af.confidence,
                agent="reviewer",
                language=af.language,
            )
        )
    return findings


def _get_max_tokens(tier: ReviewTier) -> int:
    """Get max output tokens based on review tier."""
    if tier == ReviewTier.FAST:
        return 1000
    elif tier == ReviewTier.STANDARD:
        return 2000
    else:  # DEEP
        return 3000


def _get_temperature(tier: ReviewTier) -> float:
    """Get temperature based on review tier."""
    if tier == ReviewTier.FAST:
        return 0.1  # Very deterministic for fast reviews
    return 0.2  # Slightly more creative for thorough reviews


async def run_reviewer(
    diff_context: DiffContext,
    llm: QuotaAwareFallbackLLM,
    tier: ReviewTier = ReviewTier.STANDARD,
    timeout: int = 30,
) -> list[Finding]:
    """Run the unified reviewer on a diff — single LLM call covering all categories.

    This replaces the old pattern of 4 separate specialist agent calls.
    The diff is sent exactly once, and the LLM returns findings across
    all categories (bug, security, style, test) in a single response.

    Args:
        diff_context: Parsed and redacted diff context.
        llm: QuotaAwareFallbackLLM instance.
        tier: Review depth tier (affects prompt and token limits).
        timeout: Hard timeout for the LLM call.

    Returns:
        List of validated Finding objects. Returns empty list on failure.
    """
    system_prompt = _load_system_prompt(tier)
    diff_content = _build_diff_content(diff_context)

    if not diff_content.strip():
        logger.info("reviewer.empty_diff")
        return []

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<user_diff>\n{diff_content}\n</user_diff>"},
    ]

    max_tokens = _get_max_tokens(tier)
    temperature = _get_temperature(tier)

    response = await llm.invoke(
        messages=messages,
        response_schema=ReviewerOutput,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    if response is None:
        logger.warning("reviewer.all_providers_failed")
        return []

    # Parse and validate
    try:
        output = ReviewerOutput.model_validate_json(response.content)
        findings = _to_findings(output.findings)
        logger.info(
            "reviewer.completed",
            tier=tier,
            findings=len(findings),
            model=response.model,
            provider=response.provider,
            tokens=response.tokens_used,
        )
        return findings

    except ValidationError as e:
        # Repair retry — inject error into prompt and try again
        logger.warning("reviewer.parse_error_retrying", error=str(e))
        repair_messages = messages + [
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": (
                    f"Your response failed validation. Error: {e}\n"
                    "Please try again with valid JSON matching the schema."
                ),
            },
        ]
        retry = await llm.invoke(
            messages=repair_messages,
            response_schema=ReviewerOutput,
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if retry is None:
            logger.warning("reviewer.repair_retry_failed")
            return []
        try:
            output = ReviewerOutput.model_validate_json(retry.content)
            findings = _to_findings(output.findings)
            logger.info(
                "reviewer.completed_after_repair",
                findings=len(findings),
            )
            return findings
        except Exception as e2:
            logger.error("reviewer.repair_failed", error=str(e2))
            return []
