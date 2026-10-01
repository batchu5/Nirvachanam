"""Graph node functions — bridge between LangGraph state and agent implementations.

OPTIMIZED PIPELINE (Strategy 1 + 2 + Phase 1):
  - File Selection node (deterministic — pure function, zero LLM tokens)
  - Triage node (zero LLM tokens — pure rules on pre-filtered decisions)
  - Reviewer node (SINGLE LLM call — replaces 4 specialist agents)
  - Critic node (rule-based only — no LLM)
  - Summarizer node (template-based — no LLM)

Total LLM calls: 1 (down from 6-7 in the original pipeline).

References: Token optimization §1 (single-pass), §2 (tiered review),
            new_architecture.md Phase 1 (file selection layer).
"""

from __future__ import annotations

import asyncio

import structlog

from src.agents.critic import apply_rule_based_critic
from src.agents.file_selection import (
    ExcludeReason,
    select_files,
    selected_files,
    selection_summary,
    format_selection_preview,
    total_selected_tokens,
)
from src.agents.reviewer import run_reviewer
from src.agents.triage import ReviewTier, triage_from_decisions, triage_pr
from src.llm.fallback import QuotaAwareFallbackLLM
from src.models.schemas import DiffContext, Finding, ReviewState, TokenUsage

logger = structlog.get_logger()

# Hard timeout per agent (§11)
AGENT_TIMEOUT = 30


# ---------------------------------------------------------------------------
# File Selection node (ZERO LLM tokens — deterministic pure function)
# ---------------------------------------------------------------------------

async def file_selection_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """File selection node — decides which files enter the review pipeline.

    This node is a PURE FUNCTION wrapper: no IO, no LLM calls.
    It applies the deterministic file selection logic and produces:
      1. file_decisions: full list of FileDecision objects (for triage)
      2. diff_context: filtered to only contain selected files (for reviewer)

    The original diff_context is replaced with one containing only
    the selected files — downstream nodes never see excluded files.

    Returns state update with `file_decisions` and filtered `diff_context`.
    """
    diff_context = state["diff_context"]

    # Run the pure selection function
    decisions = select_files(diff_context)

    # Log the selection preview
    summary = selection_summary(decisions)
    logger.info(
        "node.file_selection.completed",
        total_files=len(decisions),
        selected=summary.get("none", 0),
        excluded_skip_pattern=summary.get("skip_pattern", 0),
        excluded_binary=summary.get("binary", 0),
        excluded_extension=summary.get("extension", 0),
        excluded_deleted=summary.get("deleted", 0),
        excluded_too_large=summary.get("too_large", 0),
        excluded_generated=summary.get("generated", 0),
        total_selected_tokens=total_selected_tokens(decisions),
    )

    # Log detailed preview in debug mode
    logger.debug(
        "node.file_selection.preview",
        preview=format_selection_preview(decisions),
    )

    # Build a filtered DiffContext with only the selected files
    selected = selected_files(decisions)
    filtered_diff = diff_context.model_copy(update={
        "files": selected,
        "total_additions": sum(f.additions for f in selected),
        "total_deletions": sum(f.deletions for f in selected),
    })

    return {
        "file_decisions": decisions,
        "diff_context": filtered_diff,
    }


# ---------------------------------------------------------------------------
# Triage node (ZERO LLM tokens)
# ---------------------------------------------------------------------------

async def triage_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Triage node — classifies the PR into a review tier using pure rules.

    This node uses ZERO LLM tokens. When file_decisions are available
    (from the file_selection_node), it uses the new triage_from_decisions()
    for cleaner tier classification. Falls back to legacy triage_pr() if
    file_decisions are not in state.

    Returns state update with `review_tier` and `triage_reason`.
    """
    file_decisions = state.get("file_decisions")

    if file_decisions is not None:
        # Phase 1 path: triage from pre-filtered file decisions
        tier, reason = triage_from_decisions(file_decisions)
    else:
        # Legacy fallback: triage from raw diff context
        diff_context = state["diff_context"]
        tier, reason = triage_pr(diff_context)

    logger.info(
        "node.triage.completed",
        tier=tier,
        reason=reason,
        used_file_decisions=file_decisions is not None,
    )

    return {
        "review_tier": tier.value,
        "triage_reason": reason,
    }


# ---------------------------------------------------------------------------
# Unified reviewer node (SINGLE LLM call)
# ---------------------------------------------------------------------------

async def reviewer_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Unified reviewer node — single-pass review covering all categories.

    Replaces the old bug_agent, security_agent, style_agent, and test_agent
    nodes. One LLM call instead of four. The diff is sent exactly once.

    When file_selection has run, the diff_context already contains only
    selected files — the reviewer never sees excluded files.
    """
    diff_context = state["diff_context"]
    tier_str = state.get("review_tier", "standard")
    tier = ReviewTier(tier_str)

    try:
        findings = await asyncio.wait_for(
            run_reviewer(diff_context, llm, tier=tier, timeout=AGENT_TIMEOUT),
            timeout=AGENT_TIMEOUT + 5,  
        )
    except asyncio.TimeoutError:
        logger.warning("node.reviewer.timeout")
        return {"findings": [], "failed_agents": ["reviewer"]}
    except Exception as e:
        logger.error("node.reviewer.error", error=str(e))
        return {"findings": [], "failed_agents": ["reviewer"]}

    if not findings:
        return {"findings": []}

    return {"findings": findings}


async def critic_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Critic node — verifies and deduplicates findings using rule-based logic.

    This is now ALWAYS rule-based (no LLM call). The rule-based critic:
    - Drops low-confidence findings (< 0.7)
    - Detects injection echoes
    - Verifies line numbers exist in the diff
    - Deduplicates similar findings

    Zero additional LLM tokens consumed.
    """
    findings = state.get("findings", [])
    diff_context = state["diff_context"]
    tier_str = state.get("review_tier", "standard")

    if not findings:
        return {
            "verified_findings": [],
            "critic_available": True,
        }

    # For FAST tier, skip critic entirely — findings are already minimal
    if tier_str == ReviewTier.FAST:
        logger.info(
            "node.critic.skipped_fast_tier",
            findings_passthrough=len(findings),
        )
        return {
            "verified_findings": findings,
            "critic_available": True,
        }

    # Apply rule-based critic (zero LLM tokens)
    verified = apply_rule_based_critic(findings, diff_context)

    logger.info(
        "node.critic.completed",
        input_findings=len(findings),
        verified_findings=len(verified),
        critic_type="rule_based",
    )

    return {
        "verified_findings": verified,
        "critic_available": True,
    }


# ---------------------------------------------------------------------------
# Summarizer node (TEMPLATE-BASED — zero LLM tokens)
# ---------------------------------------------------------------------------

async def summarizer_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Summarizer node — produces the final PR review summary using templates.

    No LLM call needed. Formats findings into a clean markdown summary
    using the template-based approach.
    """
    verified_findings = state.get("verified_findings", [])
    diff_context = state["diff_context"]
    failed_agents = state.get("failed_agents", [])
    triage_reason = state.get("triage_reason", "")
    tier_str = state.get("review_tier", "standard")
    file_decisions = state.get("file_decisions")

    summary = _generate_summary(verified_findings, diff_context, tier_str)

    # Add file selection info if available
    if file_decisions:
        sel_summary = selection_summary(file_decisions)
        total_files = len(file_decisions)
        selected_count = sel_summary.get("none", 0)
        excluded_count = total_files - selected_count
        if excluded_count > 0:
            selection_banner = (
                f"📁 Reviewed {selected_count}/{total_files} files "
                f"({excluded_count} excluded by file selection)"
            )
            summary = f"> {selection_banner}\n\n{summary}"

    # Add triage info banner
    tier_emoji = {
        "fast": "⚡",
        "standard": "🔍",
        "deep": "🔬",
    }.get(tier_str, "🔍")
    summary = f"> {tier_emoji} **Review tier: {tier_str}** — {triage_reason}\n\n{summary}"

    # Add degraded mode banner if applicable
    if failed_agents:
        banner = (
            f"⚠️ Review completed in degraded mode — "
            f"{', '.join(failed_agents)} timed out. Results may be incomplete."
        )
        summary = f"> {banner}\n\n{summary}"

    return {"summary": summary}


def _generate_summary(
    findings: list[Finding],
    diff_context: DiffContext,
    tier: str,
) -> str:
    """Generate a template-based summary — zero LLM tokens.

    This replaces the LLM-based summarizer entirely.
    """
    if not findings:
        return (
            "## 🔍 AI Code Review Summary\n\n"
            "**Overall Assessment:** ✅ No issues detected in this PR.\n\n"
            f"Analyzed {len(diff_context.files)} files "
            f"(+{diff_context.total_additions} -{diff_context.total_deletions} lines)."
        )

    # Count by severity
    severity_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
        category_counts[str(f.category)] = category_counts.get(str(f.category), 0) + 1

    parts = ["## 🔍 AI Code Review Summary\n"]

    # Overall assessment
    critical = severity_counts.get("critical", 0)
    warnings = severity_counts.get("warning", 0)

    if critical > 0:
        parts.append(
            f"**Overall Assessment:** 🔴 Found **{critical} critical** issue(s) "
            f"that need immediate attention.\n"
        )
    elif warnings > 0:
        parts.append(
            f"**Overall Assessment:** ⚠️ Found **{warnings} warning(s)** to review.\n"
        )
    else:
        parts.append(
            f"**Overall Assessment:** ℹ️ Found **{len(findings)}** minor observation(s).\n"
        )

    # Category breakdown
    if len(category_counts) > 1:
        parts.append("### Category Breakdown\n")
        category_emoji = {
            "bug": "🐛", "security": "🔒", "style": "✨", "test": "🧪"
        }
        for cat, count in sorted(category_counts.items()):
            emoji = category_emoji.get(cat, "📋")
            parts.append(f"- {emoji} **{cat.title()}**: {count} finding(s)")
        parts.append("")

    # Key findings
    parts.append("### Key Findings\n")
    for finding in findings[:15]:
        emoji = {"critical": "🔴", "warning": "⚠️", "info": "ℹ️"}.get(
            finding.severity, "💡"
        )
        parts.append(
            f"- {emoji} **`{finding.file}:{finding.line}`** [{finding.category}] "
            f"— {finding.message}"
        )

    if len(findings) > 15:
        parts.append(
            f"\n*...and {len(findings) - 15} more findings in inline comments.*"
        )

    # Stats
    parts.append(
        f"\n### Files Reviewed\n"
        f"Analyzed {len(diff_context.files)} files "
        f"(+{diff_context.total_additions} -{diff_context.total_deletions} lines)."
    )

    return "\n".join(parts)

