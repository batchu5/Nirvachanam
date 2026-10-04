"""Tests for the token budget and cost estimation module (Phase 6).

Tests cover:
  - Per-group estimation
  - Full review cost estimation
  - Budget checking
  - Human-readable formatting
  - Edge cases (empty inputs, zero tokens, etc.)
"""

from __future__ import annotations

import pytest

from src.agents.file_selection import ExcludeReason, FileDecision
from src.agents.budget import (
    AVG_OUTPUT_TOKENS_PER_ROUND,
    AVG_ROUNDS_PER_GROUP,
    CONTEXT_INFLATION_FACTOR,
    PROMPT_OVERHEAD_TOKENS,
    BudgetCheckResult,
    GroupEstimate,
    ReviewEstimate,
    check_budget,
    estimate_group_cost,
    estimate_review_cost,
    format_estimate_banner,
)
from src.models.schemas import FileContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(
    path: str,
    *,
    additions: int = 10,
    deletions: int = 5,
) -> FileContext:
    """Create a minimal FileContext for testing."""
    lines = [f"+added line {i}" for i in range(additions)]
    lines += [f"-removed line {i}" for i in range(deletions)]
    patch = "\n".join(lines) if lines else ""
    return FileContext(
        path=path,
        patch=patch,
        additions=additions,
        deletions=deletions,
    )


def _make_decision(
    path: str,
    *,
    diff_tokens: int = 100,
    selected: bool = True,
    additions: int = 10,
    deletions: int = 5,
) -> FileDecision:
    """Create a FileDecision for testing."""
    file_ctx = _make_file(path, additions=additions, deletions=deletions)
    reason = ExcludeReason.NONE if selected else ExcludeReason.BINARY
    return FileDecision(
        file=file_ctx,
        reason=reason,
        diff_tokens=diff_tokens,
    )


# ---------------------------------------------------------------------------
# Tests for estimate_group_cost
# ---------------------------------------------------------------------------

class TestEstimateGroupCost:
    def test_basic_estimate(self):
        est = estimate_group_cost(diff_tokens=1000, file_count=3)
        assert est.diff_tokens == 1000
        assert est.files == 3
        assert est.rounds == AVG_ROUNDS_PER_GROUP
        assert est.input_tokens > 0
        assert est.output_tokens > 0
        assert est.total_tokens == est.input_tokens + est.output_tokens

    def test_output_scales_with_rounds(self):
        est_short = estimate_group_cost(diff_tokens=1000, file_count=3, rounds=3)
        est_long = estimate_group_cost(diff_tokens=1000, file_count=3, rounds=10)
        assert est_long.output_tokens > est_short.output_tokens
        assert est_long.total_tokens > est_short.total_tokens

    def test_input_includes_overhead(self):
        est = estimate_group_cost(diff_tokens=500, file_count=1, rounds=1)
        # Prompt = overhead (2000) + diff (500) = 2500
        # Input for 1 round = 2500 * inflation_factor
        expected_prompt = PROMPT_OVERHEAD_TOKENS + 500
        expected_input = int(expected_prompt * CONTEXT_INFLATION_FACTOR) * 1
        assert est.input_tokens == expected_input

    def test_zero_diff_tokens(self):
        est = estimate_group_cost(diff_tokens=0, file_count=0)
        # Still has overhead
        assert est.input_tokens > 0
        assert est.prompt_tokens == PROMPT_OVERHEAD_TOKENS

    def test_single_round(self):
        est = estimate_group_cost(diff_tokens=1000, file_count=1, rounds=1)
        assert est.rounds == 1
        assert est.output_tokens == AVG_OUTPUT_TOKENS_PER_ROUND


# ---------------------------------------------------------------------------
# Tests for estimate_review_cost
# ---------------------------------------------------------------------------

class TestEstimateReviewCost:
    def test_no_selected_files(self):
        decisions = [
            _make_decision("a.png", selected=False),
        ]
        est = estimate_review_cost(decisions)
        assert est.files == 0
        assert est.total_tokens == 0
        assert est.estimated_cost_usd == 0.0
        assert "No files" in est.estimated_cost_description

    def test_single_file_estimate(self):
        decisions = [_make_decision("main.py", diff_tokens=500)]
        est = estimate_review_cost(decisions)
        assert est.files == 1
        assert est.groups == 1
        assert est.diff_tokens == 500
        assert est.total_tokens > 0
        assert est.estimated_cost_usd >= 0

    def test_multiple_files_single_group(self):
        decisions = [
            _make_decision("a.py", diff_tokens=200),
            _make_decision("b.py", diff_tokens=300),
            _make_decision("c.py", diff_tokens=500),
        ]
        est = estimate_review_cost(decisions, num_groups=1)
        assert est.files == 3
        assert est.diff_tokens == 1000
        assert est.groups == 1

    def test_multiple_groups(self):
        decisions = [
            _make_decision("a.py", diff_tokens=500),
            _make_decision("b.py", diff_tokens=500),
        ]
        est_1group = estimate_review_cost(decisions, num_groups=1)
        est_2groups = estimate_review_cost(decisions, num_groups=2)
        # 2 groups = 2× overhead but same diff tokens
        assert est_2groups.input_tokens > est_1group.input_tokens

    def test_excludes_non_selected(self):
        decisions = [
            _make_decision("main.py", diff_tokens=500, selected=True),
            _make_decision("logo.png", diff_tokens=100, selected=False),
        ]
        est = estimate_review_cost(decisions)
        assert est.files == 1
        assert est.diff_tokens == 500

    def test_model_affects_cost(self):
        decisions = [_make_decision("main.py", diff_tokens=10_000)]
        est_flash = estimate_review_cost(decisions, model="gemini-2.5-flash")
        est_default = estimate_review_cost(decisions, model="default")
        # gemini-2.5-flash is cheaper than default
        assert est_flash.estimated_cost_usd < est_default.estimated_cost_usd

    def test_model_stored_in_estimate(self):
        decisions = [_make_decision("main.py")]
        est = estimate_review_cost(decisions, model="gemini-2.5-flash")
        assert est.model == "gemini-2.5-flash"

    def test_estimated_rounds(self):
        decisions = [_make_decision("main.py", diff_tokens=500)]
        est = estimate_review_cost(
            decisions, num_groups=2, rounds_per_group=5,
        )
        assert est.estimated_rounds == 10  # 2 groups × 5 rounds

    def test_cost_is_non_negative(self):
        decisions = [_make_decision(f"f_{i}.py", diff_tokens=1000) for i in range(10)]
        est = estimate_review_cost(decisions, num_groups=3)
        assert est.estimated_cost_usd >= 0

    def test_empty_decisions(self):
        est = estimate_review_cost([])
        assert est.files == 0
        assert est.total_tokens == 0


# ---------------------------------------------------------------------------
# Tests for check_budget
# ---------------------------------------------------------------------------

class TestCheckBudget:
    def test_within_budget(self):
        est = ReviewEstimate(
            files=5, groups=1, diff_tokens=1000,
            input_tokens=10_000, output_tokens=5_000, total_tokens=15_000,
            estimated_rounds=7, estimated_cost_usd=0.01,
            estimated_cost_description="test",
        )
        result = check_budget(est, max_total_tokens=100_000, max_cost_usd=1.0)
        assert result.within_budget is True
        assert result.recommended_action == "proceed"

    def test_over_token_budget(self):
        est = ReviewEstimate(
            files=50, groups=5, diff_tokens=100_000,
            input_tokens=400_000, output_tokens=100_000, total_tokens=500_000,
            estimated_rounds=35, estimated_cost_usd=0.50,
            estimated_cost_description="test",
        )
        result = check_budget(est, max_total_tokens=200_000, max_cost_usd=1.0)
        assert result.within_budget is False
        assert "token budget" in result.recommended_action.lower()

    def test_over_cost_budget(self):
        est = ReviewEstimate(
            files=5, groups=1, diff_tokens=1000,
            input_tokens=10_000, output_tokens=5_000, total_tokens=15_000,
            estimated_rounds=7, estimated_cost_usd=2.50,
            estimated_cost_description="test",
        )
        result = check_budget(est, max_total_tokens=100_000, max_cost_usd=1.0)
        assert result.within_budget is False
        assert "cost budget" in result.recommended_action.lower()

    def test_exactly_at_limit(self):
        est = ReviewEstimate(
            files=5, groups=1, diff_tokens=1000,
            input_tokens=80_000, output_tokens=20_000, total_tokens=100_000,
            estimated_rounds=7, estimated_cost_usd=1.0,
            estimated_cost_description="test",
        )
        result = check_budget(est, max_total_tokens=100_000, max_cost_usd=1.0)
        assert result.within_budget is True

    def test_budget_check_result_fields(self):
        est = ReviewEstimate(
            files=1, groups=1, diff_tokens=100,
            input_tokens=1000, output_tokens=500, total_tokens=1500,
            estimated_rounds=7, estimated_cost_usd=0.001,
            estimated_cost_description="test",
        )
        result = check_budget(est, max_total_tokens=50_000)
        assert result.estimate is est
        assert result.budget_tokens == 50_000


# ---------------------------------------------------------------------------
# Tests for format_estimate_banner
# ---------------------------------------------------------------------------

class TestFormatEstimateBanner:
    def test_basic_banner(self):
        est = ReviewEstimate(
            files=5, groups=2, diff_tokens=1000,
            input_tokens=42_000, output_tokens=10_000, total_tokens=52_000,
            estimated_rounds=14, estimated_cost_usd=0.003,
            estimated_cost_description="test", model="gemini-2.5-flash",
        )
        banner = format_estimate_banner(est)
        assert "💰" in banner
        assert "5 files" in banner
        assert "52,000" in banner
        assert "gemini-2.5-flash" in banner

    def test_very_cheap_cost(self):
        est = ReviewEstimate(
            files=1, groups=1, diff_tokens=100,
            input_tokens=1000, output_tokens=500, total_tokens=1500,
            estimated_rounds=7, estimated_cost_usd=0.0001,
            estimated_cost_description="test",
        )
        banner = format_estimate_banner(est)
        assert "< $0.001" in banner

    def test_moderate_cost(self):
        est = ReviewEstimate(
            files=10, groups=3, diff_tokens=10_000,
            input_tokens=100_000, output_tokens=30_000, total_tokens=130_000,
            estimated_rounds=21, estimated_cost_usd=0.05,
            estimated_cost_description="test", model="gemini-1.5-pro",
        )
        banner = format_estimate_banner(est)
        assert "$0.050" in banner or "$0.05" in banner

    def test_no_model_shows_default(self):
        est = ReviewEstimate(
            files=1, groups=1, diff_tokens=100,
            input_tokens=1000, output_tokens=500, total_tokens=1500,
            estimated_rounds=7, estimated_cost_usd=0.001,
            estimated_cost_description="test", model="",
        )
        banner = format_estimate_banner(est)
        assert "default" in banner


# ---------------------------------------------------------------------------
# Integration tests — estimate_review_cost with realistic scenarios
# ---------------------------------------------------------------------------

class TestRealisticScenarios:
    def test_small_pr(self):
        """Small PR: 2 files, ~200 diff tokens."""
        decisions = [
            _make_decision("src/utils.py", diff_tokens=100),
            _make_decision("tests/test_utils.py", diff_tokens=100),
        ]
        est = estimate_review_cost(decisions, model="gemini-2.5-flash")
        assert est.files == 2
        assert est.total_tokens < 100_000
        assert est.estimated_cost_usd < 0.10

    def test_medium_pr(self):
        """Medium PR: 10 files, ~5000 diff tokens, 3 groups."""
        decisions = [
            _make_decision(f"src/module_{i}.py", diff_tokens=500)
            for i in range(10)
        ]
        est = estimate_review_cost(
            decisions, model="gemini-2.5-flash", num_groups=3,
        )
        assert est.files == 10
        assert est.diff_tokens == 5000
        assert est.groups == 3

    def test_large_pr(self):
        """Large PR: 50 files, ~25000 diff tokens, 5 groups."""
        decisions = [
            _make_decision(f"src/file_{i}.py", diff_tokens=500)
            for i in range(50)
        ]
        est = estimate_review_cost(
            decisions, model="gemini-2.5-flash", num_groups=5,
        )
        assert est.files == 50
        assert est.groups == 5
        # Large PRs should have proportionally more tokens
        small_est = estimate_review_cost(
            [_make_decision("small.py", diff_tokens=500)],
            model="gemini-2.5-flash",
        )
        assert est.total_tokens > small_est.total_tokens
