"""Summarizer agent — produces markdown PR review summaries.

Uses Groq (primary) or Gemini (fallback) via QuotaAwareFallbackLLM.
Input: list[Finding] + DiffContext → Output: markdown summary string.

References: PRD §1b (model allocation), §2 (agent orchestration).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from src.llm.fallback import QuotaAwareFallbackLLM
from src.models.schemas import DiffContext, Finding

logger = structlog.get_logger()

# Load prompt from versioned file
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "v1" / "summarizer.md"

# JSON Schema for structured output
SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}


def _load_system_prompt() -> str:
    """Load the summarizer system prompt from the versioned prompt file."""
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    logger.warning("summarizer.prompt_file_missing", path=str(PROMPT_PATH))
    return (
        "You are a code review summarizer. Given a list of findings, produce a "
        "concise markdown summary. Return a JSON object with a 'summary' field."
    )


def _build_findings_content(
    findings: list[Finding],
    diff_context: DiffContext,
) -> str:
    """Build the input content for the summarizer.

    Includes findings as JSON + diff stats for context.
    """
    findings_json = [
        {
            "file": f.file,
            "line": f.line,
            "severity": f.severity,
            "category": f.category,
            "message": f.message,
            "suggested_fix": f.suggested_fix,
            "confidence": f.confidence,
            "agent": f.agent,
        }
        for f in findings
    ]

    stats = (
        f"PR Stats: {len(diff_context.files)} files changed, "
        f"+{diff_context.total_additions} -{diff_context.total_deletions} lines"
    )

    files_list = ", ".join(f.path for f in diff_context.files[:20])
    if len(diff_context.files) > 20:
        files_list += f" ... and {len(diff_context.files) - 20} more"

    return (
        f"{stats}\n"
        f"Files: {files_list}\n\n"
        f"<findings>\n{json.dumps(findings_json, indent=2)}\n</findings>"
    )


def _generate_fallback_summary(
    findings: list[Finding],
    diff_context: DiffContext,
) -> str:
    """Generate a simple summary without LLM if all providers fail.

    This ensures we always have SOMETHING to post, even if Groq and Gemini
    are both down.
    """
    severity_counts: dict[str, int] = {}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    parts = ["## 🔍 AI Code Review Summary\n"]

    if not findings:
        parts.append(
            "**Overall Assessment:** ✅ No issues detected in this PR.\n"
        )
        parts.append(
            f"Analyzed {len(diff_context.files)} files "
            f"(+{diff_context.total_additions} -{diff_context.total_deletions} lines)."
        )
        return "\n".join(parts)

    # Assessment
    critical = severity_counts.get("critical", 0)
    warnings = severity_counts.get("warning", 0)

    if critical > 0:
        parts.append(
            f"**Overall Assessment:** 🔴 Found {critical} critical issue(s) that need attention.\n"
        )
    elif warnings > 0:
        parts.append(
            f"**Overall Assessment:** ⚠️ Found {warnings} warning(s) to review.\n"
        )
    else:
        parts.append(
            f"**Overall Assessment:** ℹ️ Found {len(findings)} minor observation(s).\n"
        )

    # Findings summary
    parts.append("### Key Findings\n")
    for finding in findings[:10]:
        emoji = {"critical": "🔴", "warning": "⚠️", "info": "ℹ️"}.get(
            finding.severity, "💡"
        )
        parts.append(
            f"- {emoji} **{finding.file}:{finding.line}** — {finding.message}"
        )

    if len(findings) > 10:
        parts.append(f"\n*...and {len(findings) - 10} more findings in inline comments.*")

    # Stats
    parts.append(
        f"\n### Files Reviewed\n"
        f"Analyzed {len(diff_context.files)} files "
        f"(+{diff_context.total_additions} -{diff_context.total_deletions} lines)."
    )

    return "\n".join(parts)


async def run_summarizer(
    findings: list[Finding],
    diff_context: DiffContext,
    llm: QuotaAwareFallbackLLM,
    timeout: int = 30,
) -> str:
    """Run the summarizer agent to produce a PR review summary.

    Args:
        findings: List of findings from specialist agents.
        diff_context: Parsed diff context for stats.
        llm: QuotaAwareFallbackLLM instance.
        timeout: Hard timeout for the LLM call.

    Returns:
        Markdown summary string. Always returns something — falls back to
        a template-based summary if LLM fails.
    """
    system_prompt = _load_system_prompt()
    user_content = _build_findings_content(findings, diff_context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response = await llm.invoke(
        messages=messages,
        response_schema=SUMMARY_SCHEMA,
        temperature=0.4,
        max_tokens=1500,
        timeout=timeout,
    )

    if response is None:
        logger.warning("summarizer.all_providers_failed_using_fallback")
        return _generate_fallback_summary(findings, diff_context)

    # Parse the response
    try:
        data = json.loads(response.content)
        summary = data.get("summary", "")
        if summary:
            logger.info(
                "summarizer.completed",
                model=response.model,
                provider=response.provider,
                tokens=response.tokens_used,
                summary_length=len(summary),
            )
            return summary
        else:
            logger.warning("summarizer.empty_summary")
            return _generate_fallback_summary(findings, diff_context)

    except (json.JSONDecodeError, KeyError) as e:
        # If JSON parsing fails, the raw content might be a valid markdown string
        if response.content and len(response.content) > 50:
            logger.info("summarizer.using_raw_content")
            return response.content

        logger.warning("summarizer.parse_error", error=str(e))
        return _generate_fallback_summary(findings, diff_context)
