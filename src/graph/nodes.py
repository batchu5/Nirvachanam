"""Graph node functions — bridge between LangGraph state and agent implementations.

FULL OCR-STYLE PIPELINE (Phase 1 + 2 + 3 + 5 + 6):
  - File Selection node (deterministic — pure function, zero LLM tokens)
  - Triage node (zero LLM tokens — pure rules on pre-filtered decisions)
  - Rule Grouping node (deterministic — groups files by language rules)
  - Semantic Grouping node (Phase 5 — LLM or deterministic file grouping)
  - Budget Estimation node (Phase 6 — pre-run cost projection, zero LLM tokens)
  - Agent Review node (agentic tool-use loop per rule group)
  - Critic node (rule-based only — no LLM)
  - Summarizer node (template-based — no LLM)

For FAST tier PRs, the old one-shot reviewer is used instead of the
agentic loop for speed. For STANDARD/DEEP, the agent loop runs with
language-specific rules and tool access.

References: Token optimization §1 (single-pass), §2 (tiered review),
            new_architecture.md Phase 1-6.
"""

from __future__ import annotations

import asyncio

import structlog

from src.agents.agent_loop import review_group
from src.agents.budget import (
    estimate_review_cost,
    format_estimate_banner,
    check_budget,
)
from src.agents.critic import apply_rule_based_critic
from src.agents.file_selection import (
    ExcludeReason,
    select_files,
    selected_files,
    selection_summary,
    format_selection_preview,
    total_selected_tokens,
)
from src.agents.grouping import (
    FileGroup,
    group_files,
    format_grouping_summary,
    total_group_tokens,
)
from src.agents.reviewer import run_reviewer
from src.agents.triage import ReviewTier, triage_from_decisions, triage_pr
from src.llm.fallback import QuotaAwareFallbackLLM
from src.models.schemas import DiffContext, Finding, ReviewState, TokenUsage
from src.rules.engine import group_by_rules, format_rules_summary, RuleGroup

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
# Rule Grouping node (Phase 2 — deterministic, ZERO LLM tokens)
# ---------------------------------------------------------------------------

async def rule_grouping_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Rule grouping node — groups files by language-specific review rules.

    This is Phase 2 of the OCR migration. It groups selected files by
    which review rules apply to them, so that files sharing the same
    language-specific rules are reviewed together.

    The rule groups are stored in state for the agent_review_node to iterate.
    For FAST tier, this node is a no-op (the old one-shot reviewer handles it).

    ZERO LLM tokens — pure deterministic grouping.
    """
    file_decisions = state.get("file_decisions", [])
    tier_str = state.get("review_tier", "standard")

    # For FAST tier, skip grouping — the one-shot reviewer handles it
    if tier_str == ReviewTier.FAST:
        logger.info("node.rule_grouping.skipped_fast_tier")
        return {"rule_groups": []}

    # Get project rules from PR metadata if available
    # TODO: Load .reviewrules.yaml from the repository
    project_rules = None

    # Group files by rules
    rule_groups = group_by_rules(file_decisions, project_rules)

    logger.info(
        "node.rule_grouping.completed",
        total_groups=len(rule_groups),
        tier=tier_str,
        summary=format_rules_summary(rule_groups),
    )

    return {"rule_groups": rule_groups}


# ---------------------------------------------------------------------------
# Semantic Grouping node (Phase 5 — LLM or deterministic file grouping)
# ---------------------------------------------------------------------------

async def semantic_grouping_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Semantic grouping node — groups related files for batch review.

    This is Phase 5 of the OCR migration. For small PRs, all files go
    in one group (no LLM needed). For larger PRs, an LLM call groups
    files by semantic relationship using ONLY file metadata (paths,
    languages) — no diff content is sent.

    The groups are stored in state as `file_groups` for the agent review
    node to iterate over. If rule_groups already exist from Phase 2,
    they take priority and semantic grouping is skipped.

    For FAST tier, this node is a no-op.
    """
    file_decisions = state.get("file_decisions", [])
    tier_str = state.get("review_tier", "standard")
    rule_groups = state.get("rule_groups", [])

    # For FAST tier, skip grouping
    if tier_str == ReviewTier.FAST:
        logger.info("node.semantic_grouping.skipped_fast_tier")
        return {"file_groups": []}

    # If rule groups already exist and have files, skip semantic grouping
    # (Phase 2 rule grouping takes priority when available)
    if rule_groups and any(g.files for g in rule_groups):
        logger.info(
            "node.semantic_grouping.skipped_rule_groups_exist",
            rule_group_count=len(rule_groups),
        )
        return {"file_groups": []}

    # Run semantic file grouping
    file_groups = await group_files(
        file_decisions,
        llm=llm,
    )

    logger.info(
        "node.semantic_grouping.completed",
        total_groups=len(file_groups),
        total_files=sum(len(g.files) for g in file_groups),
        total_tokens=total_group_tokens(file_groups),
        tier=tier_str,
        summary=format_grouping_summary(file_groups),
    )

    return {"file_groups": file_groups}


# ---------------------------------------------------------------------------
# Budget Estimation node (Phase 6 — pre-run cost projection)
# ---------------------------------------------------------------------------

async def budget_estimation_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Budget estimation node — pre-run cost projection.

    This is Phase 6 of the OCR migration. Before any review LLM calls,
    estimate the total token cost and rough USD cost. This enables:
    - Logging for observability
    - Warnings for expensive reviews
    - Budget enforcement (skip review if over budget)

    Uses ZERO LLM tokens — pure computation on file decisions.
    """
    file_decisions = state.get("file_decisions", [])
    tier_str = state.get("review_tier", "standard")
    file_groups = state.get("file_groups", [])
    rule_groups = state.get("rule_groups", [])

    # Determine number of groups for estimation
    num_groups = len(file_groups) or len(rule_groups) or 1

    # Determine rounds per group based on tier
    if tier_str == ReviewTier.FAST:
        rounds_per_group = 1  # FAST uses one-shot reviewer
    elif tier_str == ReviewTier.DEEP:
        rounds_per_group = 10
    else:
        rounds_per_group = 7

    # Run estimation
    estimate = estimate_review_cost(
        file_decisions,
        num_groups=num_groups,
        rounds_per_group=rounds_per_group,
    )

    # Log the estimate banner
    banner = format_estimate_banner(estimate)
    logger.info(
        "node.budget_estimation.completed",
        banner=banner,
        files=estimate.files,
        groups=estimate.groups,
        total_tokens=estimate.total_tokens,
        estimated_cost_usd=estimate.estimated_cost_usd,
        tier=tier_str,
    )

    # Check against budget limits
    budget_result = check_budget(estimate)
    if not budget_result.within_budget:
        logger.warning(
            "node.budget_estimation.over_budget",
            action=budget_result.recommended_action,
        )

    # Store as dict for TypedDict compatibility
    estimate_dict = {
        "files": estimate.files,
        "groups": estimate.groups,
        "diff_tokens": estimate.diff_tokens,
        "input_tokens": estimate.input_tokens,
        "output_tokens": estimate.output_tokens,
        "total_tokens": estimate.total_tokens,
        "estimated_rounds": estimate.estimated_rounds,
        "estimated_cost_usd": estimate.estimated_cost_usd,
        "estimated_cost_description": estimate.estimated_cost_description,
        "model": estimate.model,
        "within_budget": budget_result.within_budget,
    }

    return {"review_estimate": estimate_dict}


# ---------------------------------------------------------------------------
# Agent Review node (Phase 3 — agentic tool-use loop)
# ---------------------------------------------------------------------------

async def agent_review_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Agent review node — runs the agentic tool-use loop per rule group.

    This is Phase 3 of the OCR migration. For STANDARD/DEEP tiers, it runs
    an LLM agent with tools (code_comment, file_read, code_search, etc.)
    for each rule group. The agent can read files, search the codebase,
    and post findings with verbatim code snippets.

    For FAST tier, falls back to the old one-shot reviewer for speed.

    The findings from all groups are merged via the operator.add reducer
    on the `findings` field in ReviewState.
    """
    diff_context = state["diff_context"]
    tier_str = state.get("review_tier", "standard")
    tier = ReviewTier(tier_str)
    rule_groups = state.get("rule_groups", [])
    pr_metadata = state.get("pr_metadata")

    # FAST tier: use the old one-shot reviewer
    if tier == ReviewTier.FAST or not rule_groups:
        logger.info(
            "node.agent_review.fast_path",
            tier=tier_str,
            reason="FAST tier or no rule groups",
        )
        try:
            findings = await asyncio.wait_for(
                run_reviewer(diff_context, llm, tier=tier, timeout=AGENT_TIMEOUT),
                timeout=AGENT_TIMEOUT + 5,
            )
        except asyncio.TimeoutError:
            logger.warning("node.agent_review.fast_timeout")
            return {"findings": [], "failed_agents": ["reviewer"]}
        except Exception as e:
            logger.error("node.agent_review.fast_error", error=str(e))
            return {"findings": [], "failed_agents": ["reviewer"]}

        return {"findings": findings or []}

    # STANDARD/DEEP tier: run agent loop per rule group
    all_findings: list[Finding] = []
    failed_groups: list[str] = []

    # Extract repo info for tool access
    repo = pr_metadata.repo if pr_metadata else ""
    head_sha = pr_metadata.head_sha if pr_metadata else ""
    installation_id = pr_metadata.installation_id if pr_metadata else None

    # Configure rounds based on tier
    max_rounds = 7 if tier == ReviewTier.STANDARD else 10
    budget_tokens = 50_000 if tier == ReviewTier.STANDARD else 80_000

    for group in rule_groups:
        if not group.files:
            continue

        try:
            group_findings = await asyncio.wait_for(
                review_group(
                    rule_group=group,
                    diff_context=diff_context,
                    llm=llm,
                    repo=repo,
                    head_sha=head_sha,
                    installation_id=installation_id,
                    max_rounds=max_rounds,
                    budget_tokens=budget_tokens,
                    timeout=AGENT_TIMEOUT + 15,
                ),
                timeout=(AGENT_TIMEOUT + 15) * max_rounds,
            )
            all_findings.extend(group_findings)
            logger.info(
                "node.agent_review.group_done",
                group_id=group.id,
                pattern=group.pattern,
                findings=len(group_findings),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "node.agent_review.group_timeout",
                group_id=group.id,
                pattern=group.pattern,
            )
            failed_groups.append(f"group_{group.id}_{group.pattern}")
        except Exception as e:
            logger.error(
                "node.agent_review.group_error",
                group_id=group.id,
                pattern=group.pattern,
                error=str(e),
            )
            failed_groups.append(f"group_{group.id}_{group.pattern}")

    result: dict = {"findings": all_findings}
    if failed_groups:
        result["failed_agents"] = failed_groups

    logger.info(
        "node.agent_review.completed",
        tier=tier_str,
        total_groups=len(rule_groups),
        total_findings=len(all_findings),
        failed_groups=len(failed_groups),
    )

    return result


# ---------------------------------------------------------------------------
# Legacy reviewer node (kept for backward compatibility)
# ---------------------------------------------------------------------------

async def reviewer_node(
    state: ReviewState,
    llm: QuotaAwareFallbackLLM,
) -> dict:
    """Legacy reviewer node — single-pass review covering all categories.

    This is the old one-shot reviewer, kept as a fallback for FAST tier
    and backward compatibility. For STANDARD/DEEP, use agent_review_node.
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

    # Add budget estimate banner if available (Phase 6)
    review_estimate = state.get("review_estimate")
    if review_estimate and review_estimate.get("total_tokens", 0) > 0:
        cost_usd = review_estimate.get("estimated_cost_usd", 0)
        if cost_usd < 0.001:
            cost_str = "< $0.001"
        elif cost_usd < 0.01:
            cost_str = f"~${cost_usd:.4f}"
        else:
            cost_str = f"~${cost_usd:.3f}"
        estimate_banner = (
            f"💰 **Estimated cost:** "
            f"~{review_estimate['total_tokens']:,} tokens, "
            f"{cost_str}"
        )
        summary = f"> {estimate_banner}\n\n{summary}"

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

