"""Tests for Pydantic model validation."""

import pytest
from pydantic import ValidationError

from src.models.enums import Category, Severity
from src.models.schemas import (
    DiffContext,
    FileContext,
    Finding,
    PRMetadata,
    WebhookPullRequestPayload,
    compute_content_signature,
)
from tests.conftest import SAMPLE_PR_PAYLOAD


class TestFinding:
    """Test the Finding schema validation."""

    def test_valid_finding(self):
        """A well-formed finding should pass validation."""
        f = Finding(
            file="src/auth.py",
            line=42,
            severity=Severity.WARNING,
            category=Category.BUG,
            message="Potential null dereference",
            agent="bug_agent",
        )
        assert f.file == "src/auth.py"
        assert f.line == 42
        assert f.confidence == 0.5  # default

    def test_finding_with_all_fields(self):
        """Finding with all optional fields populated."""
        f = Finding(
            file="main.py",
            line=10,
            end_line=15,
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            message="SQL injection vulnerability",
            suggested_fix="Use parameterized queries",
            confidence=0.95,
            agent="security_agent",
            language="python",
            content_signature="abc123def456",
        )
        assert f.end_line == 15
        assert f.suggested_fix is not None
        assert f.content_signature == "abc123def456"

    def test_finding_invalid_severity(self):
        """Invalid severity should fail validation."""
        with pytest.raises(ValidationError):
            Finding(
                file="x.py",
                line=1,
                severity="high",  # not in enum
                category=Category.BUG,
                message="test",
                agent="bug_agent",
            )

    def test_finding_invalid_category(self):
        """Invalid category should fail validation."""
        with pytest.raises(ValidationError):
            Finding(
                file="x.py",
                line=1,
                severity=Severity.INFO,
                category="performance",  # not in enum
                message="test",
                agent="bug_agent",
            )

    def test_finding_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            Finding(
                file="x.py",
                line=1,
                severity=Severity.INFO,
                category=Category.BUG,
                message="test",
                agent="bug_agent",
                confidence=1.5,  # out of range
            )

    def test_finding_missing_required_fields(self):
        """Missing required fields should fail validation."""
        with pytest.raises(ValidationError):
            Finding(
                file="x.py",
                # missing: line, severity, category, message, agent
            )


class TestPRMetadata:
    """Test PRMetadata model."""

    def test_minimal_metadata(self):
        """Metadata with only required fields."""
        m = PRMetadata(repo="owner/repo", pr_number=1)
        assert m.repo == "owner/repo"
        assert m.title == ""  # default

    def test_full_metadata(self):
        """Metadata with all fields."""
        m = PRMetadata(
            repo="org/project",
            pr_number=99,
            title="Fix bug",
            author="dev",
            base_branch="main",
            head_branch="fix/bug",
            installation_id=123,
            head_sha="abc",
            base_sha="def",
        )
        assert m.installation_id == 123


class TestWebhookPayload:
    """Test webhook payload parsing."""

    def test_parse_pr_payload(self):
        """Parse a real-ish GitHub PR webhook payload."""
        payload = WebhookPullRequestPayload(**SAMPLE_PR_PAYLOAD)
        assert payload.action == "opened"
        assert payload.number == 42
        assert payload.repository.full_name == "testorg/testrepo"

    def test_to_pr_metadata(self):
        """Convert webhook payload to PRMetadata."""
        payload = WebhookPullRequestPayload(**SAMPLE_PR_PAYLOAD)
        meta = payload.to_pr_metadata()

        assert meta.repo == "testorg/testrepo"
        assert meta.pr_number == 42
        assert meta.author == "testuser"
        assert meta.head_sha == "abc123def456"
        assert meta.head_branch == "fix/null-pointer"
        assert meta.base_branch == "main"

    def test_payload_without_installation(self):
        """Payload without installation field should parse fine."""
        data = {**SAMPLE_PR_PAYLOAD}
        del data["installation"]
        payload = WebhookPullRequestPayload(**data)
        meta = payload.to_pr_metadata()
        assert meta.installation_id is None


class TestDiffContext:
    """Test DiffContext model."""

    def test_empty_context(self):
        """Empty DiffContext should have sensible defaults."""
        ctx = DiffContext()
        assert len(ctx.files) == 0
        assert ctx.total_additions == 0

    def test_context_with_files(self):
        """DiffContext with file list."""
        ctx = DiffContext(
            files=[FileContext(path="a.py", additions=5, deletions=2)],
            total_additions=5,
            total_deletions=2,
            base_sha="base",
            head_sha="head",
        )
        assert len(ctx.files) == 1
        assert ctx.files[0].path == "a.py"


class TestContentSignature:
    """Test content signature computation."""

    def test_basic_signature(self):
        """Signature should be a 16-char hex string."""
        lines = ["line1", "line2", "line3", "line4", "line5"]
        sig = compute_content_signature(lines, line=3, window=2)
        assert isinstance(sig, str)
        assert len(sig) == 16

    def test_same_content_same_signature(self):
        """Same content at same position should produce same signature."""
        lines = ["a", "b", "c", "d", "e"]
        sig1 = compute_content_signature(lines, line=3, window=2)
        sig2 = compute_content_signature(lines, line=3, window=2)
        assert sig1 == sig2

    def test_different_content_different_signature(self):
        """Different content should produce different signatures."""
        lines1 = ["a", "b", "c", "d", "e"]
        lines2 = ["a", "b", "CHANGED", "d", "e"]
        sig1 = compute_content_signature(lines1, line=3, window=2)
        sig2 = compute_content_signature(lines2, line=3, window=2)
        assert sig1 != sig2

    def test_edge_case_first_line(self):
        """Signature at line 1 (near start of file) should not crash."""
        lines = ["only-line"]
        sig = compute_content_signature(lines, line=1, window=2)
        assert len(sig) == 16

    def test_edge_case_last_line(self):
        """Signature at last line should not crash."""
        lines = ["a", "b", "c"]
        sig = compute_content_signature(lines, line=3, window=2)
        assert len(sig) == 16
