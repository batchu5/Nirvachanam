"""Tests for the run identity and checkpoint resume module (Phase 7).

Tests cover:
  - Deterministic identity hash computation
  - File order independence
  - Same inputs → same hash
  - Different inputs → different hash
  - Identity from FileDecision objects
  - Formatting helpers
  - RunCheckResult actions
"""

from __future__ import annotations

import pytest

from src.agents.file_selection import ExcludeReason, FileDecision
from src.graph.run_identity import (
    RunCheckResult,
    RunIdentity,
    RunStatus,
    compute_run_identity,
    compute_run_identity_from_decisions,
    format_run_identity,
    format_skip_message,
)
from src.models.schemas import FileContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(path: str) -> FileContext:
    """Create a minimal FileContext for testing."""
    return FileContext(
        path=path,
        patch=f"+line in {path}",
        additions=1,
        deletions=0,
    )


def _make_decision(
    path: str,
    *,
    selected: bool = True,
    diff_tokens: int = 100,
) -> FileDecision:
    """Create a FileDecision for testing."""
    file_ctx = _make_file(path)
    reason = ExcludeReason.NONE if selected else ExcludeReason.BINARY
    return FileDecision(
        file=file_ctx,
        reason=reason,
        diff_tokens=diff_tokens,
    )


# ---------------------------------------------------------------------------
# Tests for compute_run_identity
# ---------------------------------------------------------------------------

class TestComputeRunIdentity:
    def test_basic_identity(self):
        identity = compute_run_identity(
            selected_files=["src/main.py", "src/utils.py"],
            head_sha="abc123def456",
            base_sha="main789xyz",
        )
        assert identity.identity_hash  # non-empty
        assert len(identity.identity_hash) == 64  # SHA-256 hex digest
        assert identity.short_hash == identity.identity_hash[:12]
        assert identity.head_sha == "abc123def456"
        assert identity.base_sha == "main789xyz"
        assert identity.file_count == 2
        assert identity.selected_files == ["src/main.py", "src/utils.py"]

    def test_deterministic(self):
        """Same inputs must produce same hash."""
        id1 = compute_run_identity(["a.py", "b.py"], "sha1", "sha2")
        id2 = compute_run_identity(["a.py", "b.py"], "sha1", "sha2")
        assert id1.identity_hash == id2.identity_hash

    def test_order_independent(self):
        """File order must not affect the hash."""
        id1 = compute_run_identity(["b.py", "a.py", "c.py"], "sha1", "sha2")
        id2 = compute_run_identity(["c.py", "a.py", "b.py"], "sha1", "sha2")
        assert id1.identity_hash == id2.identity_hash
        # Files should be stored sorted
        assert id1.selected_files == ["a.py", "b.py", "c.py"]

    def test_different_files_different_hash(self):
        id1 = compute_run_identity(["a.py"], "sha1", "sha2")
        id2 = compute_run_identity(["b.py"], "sha1", "sha2")
        assert id1.identity_hash != id2.identity_hash

    def test_different_head_sha_different_hash(self):
        id1 = compute_run_identity(["a.py"], "sha1", "base")
        id2 = compute_run_identity(["a.py"], "sha2", "base")
        assert id1.identity_hash != id2.identity_hash

    def test_different_base_sha_different_hash(self):
        id1 = compute_run_identity(["a.py"], "head", "base1")
        id2 = compute_run_identity(["a.py"], "head", "base2")
        assert id1.identity_hash != id2.identity_hash

    def test_empty_files(self):
        identity = compute_run_identity([], "sha1", "sha2")
        assert identity.file_count == 0
        assert identity.selected_files == []
        assert identity.identity_hash  # still produces a valid hash

    def test_empty_shas(self):
        identity = compute_run_identity(["a.py"], "", "")
        assert identity.identity_hash
        assert identity.head_sha == ""
        assert identity.base_sha == ""


# ---------------------------------------------------------------------------
# Tests for RunIdentity.as_thread_id
# ---------------------------------------------------------------------------

class TestAsThreadId:
    def test_thread_id_format(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        thread_id = identity.as_thread_id()
        assert thread_id.startswith("review-")
        assert identity.short_hash in thread_id

    def test_same_identity_same_thread_id(self):
        id1 = compute_run_identity(["a.py"], "sha1", "sha2")
        id2 = compute_run_identity(["a.py"], "sha1", "sha2")
        assert id1.as_thread_id() == id2.as_thread_id()

    def test_different_identity_different_thread_id(self):
        id1 = compute_run_identity(["a.py"], "sha1", "sha2")
        id2 = compute_run_identity(["b.py"], "sha1", "sha2")
        assert id1.as_thread_id() != id2.as_thread_id()


# ---------------------------------------------------------------------------
# Tests for compute_run_identity_from_decisions
# ---------------------------------------------------------------------------

class TestComputeFromDecisions:
    def test_only_selected_files(self):
        decisions = [
            _make_decision("src/main.py", selected=True),
            _make_decision("logo.png", selected=False),
            _make_decision("src/utils.py", selected=True),
        ]
        identity = compute_run_identity_from_decisions(
            decisions, "sha1", "sha2",
        )
        assert identity.file_count == 2
        assert "logo.png" not in identity.selected_files
        assert "src/main.py" in identity.selected_files
        assert "src/utils.py" in identity.selected_files

    def test_matches_manual_computation(self):
        """Identity from decisions should match manual computation."""
        decisions = [
            _make_decision("a.py", selected=True),
            _make_decision("b.py", selected=True),
        ]
        id_from_decisions = compute_run_identity_from_decisions(
            decisions, "sha1", "sha2",
        )
        id_manual = compute_run_identity(
            ["a.py", "b.py"], "sha1", "sha2",
        )
        assert id_from_decisions.identity_hash == id_manual.identity_hash

    def test_no_selected_files(self):
        decisions = [
            _make_decision("a.png", selected=False),
        ]
        identity = compute_run_identity_from_decisions(
            decisions, "sha1", "sha2",
        )
        assert identity.file_count == 0

    def test_empty_decisions(self):
        identity = compute_run_identity_from_decisions([], "sha1", "sha2")
        assert identity.file_count == 0
        assert identity.identity_hash  # valid hash even with no files


# ---------------------------------------------------------------------------
# Tests for RunCheckResult
# ---------------------------------------------------------------------------

class TestRunCheckResult:
    def test_run_action(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        result = RunCheckResult(
            identity=identity,
            status=RunStatus.NOT_FOUND,
            action="run",
        )
        assert result.action == "run"
        assert result.review_id is None

    def test_skip_action(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        result = RunCheckResult(
            identity=identity,
            status=RunStatus.COMPLETED,
            review_id="abc-123",
            summary="Previous completed",
            action="skip",
        )
        assert result.action == "skip"
        assert result.review_id == "abc-123"

    def test_resume_action(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        result = RunCheckResult(
            identity=identity,
            status=RunStatus.PARTIAL,
            review_id="abc-456",
            action="resume",
        )
        assert result.action == "resume"

    def test_retry_action(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        result = RunCheckResult(
            identity=identity,
            status=RunStatus.FAILED,
            review_id="abc-789",
            action="retry",
        )
        assert result.action == "retry"


# ---------------------------------------------------------------------------
# Tests for format_run_identity
# ---------------------------------------------------------------------------

class TestFormatRunIdentity:
    def test_basic_format(self):
        identity = compute_run_identity(
            ["a.py", "b.py", "c.py"],
            "abc123def456",
            "main789xyz",
        )
        formatted = format_run_identity(identity)
        assert "🔑" in formatted
        assert identity.short_hash in formatted
        assert "3 files" in formatted
        assert "abc123d" in formatted  # head_sha[:7]
        assert "main789" in formatted  # base_sha[:7]

    def test_empty_shas(self):
        identity = compute_run_identity(["a.py"], "", "")
        formatted = format_run_identity(identity)
        assert "?" in formatted  # fallback for empty SHA


# ---------------------------------------------------------------------------
# Tests for format_skip_message
# ---------------------------------------------------------------------------

class TestFormatSkipMessage:
    def test_skip_message(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        result = RunCheckResult(
            identity=identity,
            status=RunStatus.COMPLETED,
            summary="Previous review completed",
            action="skip",
        )
        msg = format_skip_message(result)
        assert "⏭️" in msg
        assert "Skipping" in msg
        assert identity.short_hash in msg

    def test_resume_message(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        result = RunCheckResult(
            identity=identity,
            status=RunStatus.PARTIAL,
            action="resume",
        )
        msg = format_skip_message(result)
        assert "🔄" in msg
        assert "Resuming" in msg

    def test_retry_message(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        result = RunCheckResult(
            identity=identity,
            status=RunStatus.FAILED,
            action="retry",
        )
        msg = format_skip_message(result)
        assert "🔄" in msg
        assert "Retrying" in msg

    def test_run_message_empty(self):
        identity = compute_run_identity(["a.py"], "sha1", "sha2")
        result = RunCheckResult(
            identity=identity,
            status=RunStatus.NOT_FOUND,
            action="run",
        )
        msg = format_skip_message(result)
        assert msg == ""
