"""Token budget and cost estimation — pre-run cost projection (Phase 6).

Before any LLM calls, this module estimates the total token cost of a
review run. The estimate accounts for:
  - Prompt overhead (system prompt + tool definitions + instructions)
  - Diff token cost per file (estimated from patch line count)
  - Agent loop inflation (each round adds tool results to the context)
  - Average output tokens per round

The estimate is deliberately conservative — overestimation is safer
than under-estimation. The output is a ReviewEstimate that can be:
  - Logged for observability
  - Used to warn users about expensive reviews
  - Fed back into grouping to split oversized groups
  - Compared to actual usage post-run for calibration

References: new_architecture.md Phase 6, Alibaba OCR estimate.go
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.agents.file_selection import FileDecision

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Cost model constants (empirical, conservative)
# ---------------------------------------------------------------------------

# Fixed overhead per LLM call: system prompt + tool definitions + instructions
PROMPT_OVERHEAD_TOKENS = 2_000

# Average number of agent loop rounds per review group
AVG_ROUNDS_PER_GROUP = 7

# Average output tokens per agent loop round (tool calls + reasoning)
AVG_OUTPUT_TOKENS_PER_ROUND = 700

# Average tool result size fed back into context per round
AVG_TOOL_RESULT_TOKENS_PER_ROUND = 300

# Agent loop inflation factor: each round adds to the context window,
# so later rounds process more input tokens than earlier ones.
# This is the average multiplier across all rounds.
CONTEXT_INFLATION_FACTOR = 1.5

# Rough cost per 1M tokens for various models (USD, for display only)
MODEL_COST_PER_MILLION: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "default": {"input": 0.50, "output": 1.00},
}


# ---------------------------------------------------------------------------
# ReviewEstimate — the output of estimate_review_cost()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewEstimate:
    """Pre-run, order-of-magnitude cost projection.

    Attributes:
        files: Number of files that will be reviewed.
        groups: Number of review groups.
        diff_tokens: Total estimated diff tokens across all files.
        input_tokens: Estimated total input tokens (all rounds, all groups).
        output_tokens: Estimated total output tokens.
        total_tokens: input_tokens + output_tokens.
        estimated_rounds: Total estimated agent loop rounds.
        estimated_cost_usd: Rough USD cost estimate (for display only).
        estimated_cost_description: Human-readable cost description.
        model: Model name used for cost lookup.
    """
    files: int
    groups: int
    diff_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_rounds: int
    estimated_cost_usd: float
    estimated_cost_description: str
    model: str = ""


# ---------------------------------------------------------------------------
# Per-group estimation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupEstimate:
    """Token estimate for a single review group."""
    group_id: int
    files: int
    diff_tokens: int
    prompt_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    rounds: int


def estimate_group_cost(
    diff_tokens: int,
    file_count: int,
    *,
    rounds: int = AVG_ROUNDS_PER_GROUP,
) -> GroupEstimate:
    """Estimate token cost for a single review group.

    The model: each round the LLM sees:
      - Round 1: prompt_overhead + diff_tokens → output_tokens
      - Round N: prompt_overhead + diff_tokens + sum(prev_tool_results) → output_tokens

    We approximate this with an inflation factor that accounts for
    growing context across rounds.

    Args:
        diff_tokens: Estimated diff tokens for all files in the group.
        file_count: Number of files in the group (affects round count).
        rounds: Expected number of agent loop rounds.

    Returns:
        GroupEstimate with token projections.
    """
    # Prompt = system prompt overhead + diff content
    prompt_tokens = PROMPT_OVERHEAD_TOKENS + diff_tokens

    # Input: prompt sent each round, plus accumulated tool results
    # Round 1: prompt_tokens
    # Round 2: prompt_tokens + tool_result_1
    # Round N: prompt_tokens + tool_result_1..N-1
    # Average input per round ≈ prompt_tokens * inflation_factor
    avg_input_per_round = int(prompt_tokens * CONTEXT_INFLATION_FACTOR)
    total_input = avg_input_per_round * rounds

    # Output: each round produces tool calls and reasoning
    total_output = AVG_OUTPUT_TOKENS_PER_ROUND * rounds

    return GroupEstimate(
        group_id=0,
        files=file_count,
        diff_tokens=diff_tokens,
        prompt_tokens=prompt_tokens,
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        rounds=rounds,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_review_cost(
    decisions: list[FileDecision],
    *,
    model: str = "",
    num_groups: int = 1,
    rounds_per_group: int = AVG_ROUNDS_PER_GROUP,
) -> ReviewEstimate:
    """Pre-run, order-of-magnitude cost projection.

    Estimates total token usage and rough cost before any LLM calls.
    The estimate is conservative — actual usage will typically be lower.

    Args:
        decisions: List of FileDecision objects from select_files().
            Only selected files are counted.
        model: Optional model name for cost lookup.
        num_groups: Expected number of review groups (from grouping).
        rounds_per_group: Expected agent loop rounds per group.

    Returns:
        ReviewEstimate with token and cost projections.
    """
    # Filter to only selected files
    selected = [d for d in decisions if d.selected]

    if not selected:
        return ReviewEstimate(
            files=0,
            groups=0,
            diff_tokens=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_rounds=0,
            estimated_cost_usd=0.0,
            estimated_cost_description="No files to review",
            model=model,
        )

    total_diff_tokens = sum(d.diff_tokens for d in selected)
    file_count = len(selected)

    # If num_groups is 1 (default), estimate based on file count
    effective_groups = max(num_groups, 1)

    # Split diff tokens roughly equally across groups
    tokens_per_group = total_diff_tokens // effective_groups if effective_groups > 0 else total_diff_tokens
    files_per_group = file_count // effective_groups if effective_groups > 0 else file_count

    # Estimate each group
    total_input = 0
    total_output = 0
    total_rounds = 0

    for _ in range(effective_groups):
        group_est = estimate_group_cost(
            diff_tokens=tokens_per_group,
            file_count=files_per_group,
            rounds=rounds_per_group,
        )
        total_input += group_est.input_tokens
        total_output += group_est.output_tokens
        total_rounds += group_est.rounds

    total_tokens = total_input + total_output

    # Cost estimate
    cost_rates = MODEL_COST_PER_MILLION.get(
        model, MODEL_COST_PER_MILLION["default"]
    )
    input_cost = (total_input / 1_000_000) * cost_rates["input"]
    output_cost = (total_output / 1_000_000) * cost_rates["output"]
    total_cost = input_cost + output_cost

    # Human-readable cost description
    cost_desc = _format_cost_description(
        file_count=file_count,
        groups=effective_groups,
        total_tokens=total_tokens,
        total_cost=total_cost,
        model=model,
    )

    estimate = ReviewEstimate(
        files=file_count,
        groups=effective_groups,
        diff_tokens=total_diff_tokens,
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_tokens,
        estimated_rounds=total_rounds,
        estimated_cost_usd=round(total_cost, 6),
        estimated_cost_description=cost_desc,
        model=model,
    )

    logger.info(
        "budget.estimate",
        files=estimate.files,
        groups=estimate.groups,
        diff_tokens=estimate.diff_tokens,
        input_tokens=estimate.input_tokens,
        output_tokens=estimate.output_tokens,
        total_tokens=estimate.total_tokens,
        estimated_rounds=estimate.estimated_rounds,
        estimated_cost_usd=estimate.estimated_cost_usd,
        model=model,
    )

    return estimate


# ---------------------------------------------------------------------------
# Cost description formatter
# ---------------------------------------------------------------------------

def _format_cost_description(
    *,
    file_count: int,
    groups: int,
    total_tokens: int,
    total_cost: float,
    model: str,
) -> str:
    """Format a human-readable cost description."""
    model_label = model or "default model"

    if total_cost < 0.001:
        cost_str = "< $0.001"
    elif total_cost < 0.01:
        cost_str = f"~${total_cost:.4f}"
    elif total_cost < 1.0:
        cost_str = f"~${total_cost:.3f}"
    else:
        cost_str = f"~${total_cost:.2f}"

    # Token tier description
    if total_tokens < 10_000:
        tier = "lightweight"
    elif total_tokens < 50_000:
        tier = "standard"
    elif total_tokens < 200_000:
        tier = "heavy"
    else:
        tier = "very heavy"

    parts = [
        f"{file_count} files in {groups} group(s)",
        f"~{total_tokens:,} total tokens ({tier})",
        f"~{groups * AVG_ROUNDS_PER_GROUP} agent rounds",
        f"{cost_str} estimated ({model_label})",
    ]

    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Budget guard — check if review is within budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetCheckResult:
    """Result of checking a review estimate against a budget.

    Attributes:
        within_budget: True if the estimated cost is within budget.
        estimate: The ReviewEstimate that was checked.
        budget_tokens: The maximum allowed tokens.
        recommended_action: Suggested action if over budget.
    """
    within_budget: bool
    estimate: ReviewEstimate
    budget_tokens: int
    recommended_action: str


def check_budget(
    estimate: ReviewEstimate,
    *,
    max_total_tokens: int = 500_000,
    max_cost_usd: float = 1.0,
) -> BudgetCheckResult:
    """Check whether a review estimate is within budget.

    Args:
        estimate: Pre-run cost estimate.
        max_total_tokens: Maximum total tokens allowed.
        max_cost_usd: Maximum USD cost allowed.

    Returns:
        BudgetCheckResult indicating whether the review is within budget
        and suggesting actions if it's not.
    """
    within_budget = (
        estimate.total_tokens <= max_total_tokens
        and estimate.estimated_cost_usd <= max_cost_usd
    )

    if within_budget:
        action = "proceed"
    elif estimate.total_tokens > max_total_tokens:
        overage = estimate.total_tokens - max_total_tokens
        pct = (overage / max_total_tokens) * 100
        action = (
            f"Over token budget by ~{overage:,} tokens ({pct:.0f}%). "
            f"Consider reducing review depth, increasing groups, "
            f"or excluding low-priority files."
        )
    else:
        action = (
            f"Over cost budget (${estimate.estimated_cost_usd:.4f} > "
            f"${max_cost_usd:.2f}). Consider using a cheaper model "
            f"or reducing review scope."
        )

    result = BudgetCheckResult(
        within_budget=within_budget,
        estimate=estimate,
        budget_tokens=max_total_tokens,
        recommended_action=action,
    )

    if not within_budget:
        logger.warning(
            "budget.over_budget",
            total_tokens=estimate.total_tokens,
            budget=max_total_tokens,
            cost_usd=estimate.estimated_cost_usd,
            max_cost=max_cost_usd,
            action=action,
        )

    return result


# ---------------------------------------------------------------------------
# Convenience: format estimate as a banner for the review summary
# ---------------------------------------------------------------------------

def format_estimate_banner(estimate: ReviewEstimate) -> str:
    """Format a concise banner string for the review summary.

    Example: "💰 Estimated: 5 files, ~42,000 tokens, ~$0.003 (gemini-2.5-flash)"
    """
    model_label = estimate.model or "default"

    if estimate.estimated_cost_usd < 0.001:
        cost_str = "< $0.001"
    elif estimate.estimated_cost_usd < 0.01:
        cost_str = f"~${estimate.estimated_cost_usd:.4f}"
    else:
        cost_str = f"~${estimate.estimated_cost_usd:.3f}"

    return (
        f"💰 Estimated: {estimate.files} files, "
        f"~{estimate.total_tokens:,} tokens, "
        f"{cost_str} ({model_label})"
    )
