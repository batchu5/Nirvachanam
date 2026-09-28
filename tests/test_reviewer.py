"""Tests for GitHub review builder — line-to-position mapping and formatting."""

from __future__ import annotations

import pytest

from src.github.reviewer import (
    _build_position_map,
    build_review_body,
    build_review_comments,
)
from src.models.schemas import DiffContext, FileContext, Finding, HunkInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_diff_context():
    """A DiffContext with one file and one hunk."""
    return DiffContext(
        files=[
            FileContext(
                path="src/utils.py",
                language="python",
                hunks=[
                    HunkInfo(
                        source_start=10,
                        source_length=3,
                        target_start=10,
                        target_length=5,
                        added_lines=[
                            (12, "    new_line_1 = True"),
                            (13, "    new_line_2 = False"),
                        ],
                        removed_lines=[],
                    ),
                ],
                patch="...",
                additions=2,
                deletions=0,
            ),
        ],
        total_additions=2,
        total_deletions=0,
    )


@pytest.fixture
def sample_findings():
    """A list of findings for testing."""
    return [
        Finding(
            file="src/utils.py",
            line=12,
            severity="warning",
            category="bug",
            message="Variable 'new_line_1' is always True, consider using a constant.",
            confidence=0.7,
            agent="bug_agent",
            language="python",
        ),
        Finding(
            file="src/utils.py",
            line=13,
            severity="info",
            category="bug",
            message="Consider adding type hints.",
            confidence=0.4,
            agent="bug_agent",
            language="python",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildPositionMap:
    """Tests for position map construction."""

    def test_builds_positions_for_added_lines(self, simple_diff_context):
        pos_map = _build_position_map(simple_diff_context)
        assert "src/utils.py" in pos_map
        # Added lines should have positions
        file_positions = pos_map["src/utils.py"]
        assert 12 in file_positions
        assert 13 in file_positions

    def test_empty_diff_returns_empty(self):
        empty_ctx = DiffContext(files=[], total_additions=0, total_deletions=0)
        pos_map = _build_position_map(empty_ctx)
        assert pos_map == {}

    def test_positions_are_sequential(self, simple_diff_context):
        pos_map = _build_position_map(simple_diff_context)
        positions = pos_map["src/utils.py"]
        # Position for line 12 should be before position for line 13
        if 12 in positions and 13 in positions:
            assert positions[12] < positions[13]


class TestBuildReviewComments:
    """Tests for review comment construction."""

    def test_builds_comments_for_mapped_findings(
        self, sample_findings, simple_diff_context
    ):
        comments = build_review_comments(sample_findings, simple_diff_context)
        # At least some findings should map successfully
        assert len(comments) >= 1
        # Each comment should have required fields
        for comment in comments:
            assert "path" in comment
            assert "position" in comment
            assert "body" in comment

    def test_comment_body_includes_severity(
        self, sample_findings, simple_diff_context
    ):
        comments = build_review_comments(sample_findings, simple_diff_context)
        if comments:
            assert "WARNING" in comments[0]["body"] or "INFO" in comments[0]["body"]

    def test_unmapped_findings_skipped(self):
        """Findings for files not in the diff should be skipped."""
        findings = [
            Finding(
                file="nonexistent.py",
                line=5,
                severity="warning",
                category="bug",
                message="some issue",
                confidence=0.5,
                agent="bug_agent",
            ),
        ]
        empty_ctx = DiffContext(files=[], total_additions=0, total_deletions=0)
        comments = build_review_comments(findings, empty_ctx)
        assert comments == []


class TestBuildReviewBody:
    """Tests for review body construction."""

    def test_includes_summary(self, sample_findings):
        body = build_review_body("## Summary\nLooks good!", sample_findings)
        assert "Summary" in body
        assert "Looks good!" in body

    def test_includes_stats(self, sample_findings):
        body = build_review_body("summary", sample_findings)
        assert "2 findings" in body

    def test_degraded_mode_banner(self, sample_findings):
        body = build_review_body(
            "summary", sample_findings, degraded_agents=["security_agent"]
        )
        assert "degraded" in body.lower()
        assert "security_agent" in body

    def test_no_degraded_banner_when_clean(self, sample_findings):
        body = build_review_body("summary", sample_findings)
        assert "degraded" not in body.lower()

    def test_includes_bot_signature(self, sample_findings):
        body = build_review_body("summary", sample_findings)
        assert "AI PR Review Agent" in body

    def test_empty_findings(self):
        body = build_review_body("All good!", [])
        assert "All good!" in body
