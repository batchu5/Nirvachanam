"""Tests for the deterministic file selection layer (Phase 1).

Tests the pure function select_files() and all its helper functions
across a variety of scenarios: binary files, skip patterns, extension
allowlists, generated code detection, deleted files, oversized diffs,
and the convenience helper functions.
"""

from __future__ import annotations

import pytest

from src.agents.file_selection import (
    ExcludeReason,
    FileDecision,
    select_files,
    selected_files,
    retained_files,
    total_selected_tokens,
    selection_summary,
    format_selection_preview,
    DEFAULT_ALLOWED_EXTENSIONS,
    DEFAULT_SKIP_PATTERNS,
    _get_extension,
    _is_binary_extension,
    _matches_skip_pattern,
    _has_allowed_extension,
    _is_generated_code,
    _estimate_diff_tokens,
)
from src.models.schemas import DiffContext, FileContext, HunkInfo


# ---------------------------------------------------------------------------
# Helpers to build test data
# ---------------------------------------------------------------------------

def _make_file(
    path: str,
    *,
    additions: int = 10,
    deletions: int = 5,
    is_deleted: bool = False,
    is_new: bool = False,
    patch: str = "",
    language: str | None = None,
) -> FileContext:
    """Create a minimal FileContext for testing."""
    if not patch:
        # Generate a simple patch with the right number of lines
        lines = [f"+added line {i}" for i in range(additions)]
        lines += [f"-removed line {i}" for i in range(deletions)]
        patch = "\n".join(lines) if lines else ""
    return FileContext(
        path=path,
        language=language,
        is_new=is_new,
        is_deleted=is_deleted,
        patch=patch,
        additions=additions,
        deletions=deletions,
    )


def _make_diff(*files: FileContext) -> DiffContext:
    """Create a DiffContext from a list of FileContext objects."""
    return DiffContext(
        files=list(files),
        total_additions=sum(f.additions for f in files),
        total_deletions=sum(f.deletions for f in files),
    )


# ---------------------------------------------------------------------------
# Tests for _get_extension
# ---------------------------------------------------------------------------

class TestGetExtension:
    def test_python_file(self):
        assert _get_extension("src/utils/helpers.py") == ".py"

    def test_typescript_file(self):
        assert _get_extension("app/index.tsx") == ".tsx"

    def test_no_extension(self):
        assert _get_extension("Makefile") == ""

    def test_dotfile(self):
        assert _get_extension(".gitignore") == ".gitignore"

    def test_compound_extension(self):
        # Returns last extension
        assert _get_extension("app.test.ts") == ".ts"

    def test_deeply_nested(self):
        assert _get_extension("a/b/c/d/e.go") == ".go"


# ---------------------------------------------------------------------------
# Tests for _is_binary_extension
# ---------------------------------------------------------------------------

class TestIsBinaryExtension:
    def test_png_is_binary(self):
        assert _is_binary_extension("logo.png") is True

    def test_jpg_is_binary(self):
        assert _is_binary_extension("photo.jpg") is True

    def test_py_is_not_binary(self):
        assert _is_binary_extension("main.py") is False

    def test_woff_is_binary(self):
        assert _is_binary_extension("font.woff2") is True

    def test_exe_is_binary(self):
        assert _is_binary_extension("app.exe") is True


# ---------------------------------------------------------------------------
# Tests for _matches_skip_pattern
# ---------------------------------------------------------------------------

class TestMatchesSkipPattern:
    def test_lock_file_matches(self):
        result = _matches_skip_pattern("package-lock.json", DEFAULT_SKIP_PATTERNS)
        assert result is not None

    def test_min_js_matches(self):
        result = _matches_skip_pattern("bundle.min.js", DEFAULT_SKIP_PATTERNS)
        assert result == "*.min.js"

    def test_vendor_matches(self):
        result = _matches_skip_pattern("vendor/lib/foo.go", DEFAULT_SKIP_PATTERNS)
        assert result == "vendor/*"

    def test_node_modules_matches(self):
        result = _matches_skip_pattern("node_modules/react/index.js", DEFAULT_SKIP_PATTERNS)
        assert result == "node_modules/*"

    def test_normal_file_no_match(self):
        result = _matches_skip_pattern("src/main.py", DEFAULT_SKIP_PATTERNS)
        assert result is None

    def test_pycache_matches(self):
        result = _matches_skip_pattern("__pycache__/foo.pyc", DEFAULT_SKIP_PATTERNS)
        assert result is not None

    def test_custom_pattern(self):
        result = _matches_skip_pattern(
            "internal/generated.go",
            ["internal/*"],
        )
        assert result == "internal/*"


# ---------------------------------------------------------------------------
# Tests for _has_allowed_extension
# ---------------------------------------------------------------------------

class TestHasAllowedExtension:
    def test_python_allowed(self):
        assert _has_allowed_extension("main.py", DEFAULT_ALLOWED_EXTENSIONS) is True

    def test_typescript_allowed(self):
        assert _has_allowed_extension("app.tsx", DEFAULT_ALLOWED_EXTENSIONS) is True

    def test_unknown_extension_not_allowed(self):
        assert _has_allowed_extension("data.xyz", DEFAULT_ALLOWED_EXTENSIONS) is False

    def test_none_allows_all(self):
        assert _has_allowed_extension("anything.xyz", None) is True

    def test_no_extension_not_allowed(self):
        assert _has_allowed_extension("Makefile", DEFAULT_ALLOWED_EXTENSIONS) is False


# ---------------------------------------------------------------------------
# Tests for _is_generated_code
# ---------------------------------------------------------------------------

class TestIsGeneratedCode:
    def test_generated_marker_detected(self):
        patch = "// Code generated by protoc-gen-go. DO NOT EDIT.\npackage pb"
        assert _is_generated_code(patch) is True

    def test_auto_generated_marker(self):
        patch = "# Auto-generated by alembic\nrevision = '12345'"
        assert _is_generated_code(patch) is True

    def test_at_generated_marker(self):
        patch = "// @generated\nexport const schema = {};"
        assert _is_generated_code(patch) is True

    def test_normal_code_not_generated(self):
        patch = "def hello():\n    return 'world'"
        assert _is_generated_code(patch) is False

    def test_empty_patch(self):
        assert _is_generated_code("") is False


# ---------------------------------------------------------------------------
# Tests for _estimate_diff_tokens
# ---------------------------------------------------------------------------

class TestEstimateDiffTokens:
    def test_empty_patch(self):
        file_ctx = _make_file("test.py", patch="", additions=0, deletions=0)
        assert _estimate_diff_tokens(file_ctx) == 0

    def test_single_line(self):
        file_ctx = _make_file("test.py", patch="+added line")
        assert _estimate_diff_tokens(file_ctx) == 4  # 1 line × 4 tokens/line

    def test_multi_line(self):
        lines = "\n".join([f"+line {i}" for i in range(10)])
        file_ctx = _make_file("test.py", patch=lines)
        assert _estimate_diff_tokens(file_ctx) == 40  # 10 lines × 4


# ---------------------------------------------------------------------------
# Tests for select_files — the main pure function
# ---------------------------------------------------------------------------

class TestSelectFiles:
    def test_empty_diff(self):
        diff = _make_diff()
        decisions = select_files(diff)
        assert decisions == []

    def test_single_python_file_selected(self):
        diff = _make_diff(_make_file("src/main.py"))
        decisions = select_files(diff)
        assert len(decisions) == 1
        assert decisions[0].selected is True
        assert decisions[0].reason == ExcludeReason.NONE

    def test_binary_file_excluded(self):
        diff = _make_diff(_make_file("logo.png"))
        decisions = select_files(diff)
        assert len(decisions) == 1
        assert decisions[0].selected is False
        assert decisions[0].reason == ExcludeReason.BINARY

    def test_skip_pattern_excluded(self):
        diff = _make_diff(_make_file("package-lock.json"))
        decisions = select_files(diff)
        assert len(decisions) == 1
        assert decisions[0].selected is False
        assert decisions[0].reason == ExcludeReason.SKIP_PATTERN

    def test_extension_not_in_allowlist(self):
        diff = _make_diff(_make_file("data.xyz"))
        decisions = select_files(diff)
        assert len(decisions) == 1
        assert decisions[0].selected is False
        assert decisions[0].reason == ExcludeReason.EXTENSION

    def test_deleted_file_retained_not_selected(self):
        diff = _make_diff(_make_file("old.py", is_deleted=True))
        decisions = select_files(diff)
        assert len(decisions) == 1
        assert decisions[0].selected is False
        assert decisions[0].retained is True
        assert decisions[0].reason == ExcludeReason.DELETED

    def test_generated_code_excluded(self):
        patch = "// Code generated by protoc. DO NOT EDIT.\npackage pb\n" + "+line\n" * 5
        diff = _make_diff(_make_file("service.pb.go", patch=patch))
        # Note: .pb.go also matches skip pattern, but we test with skip_patterns=[]
        decisions = select_files(diff, skip_patterns=[])
        assert len(decisions) == 1
        assert decisions[0].reason == ExcludeReason.GENERATED

    def test_too_large_file_excluded(self):
        # Create a file with a very large patch (>4000 tokens)
        large_patch = "\n".join([f"+{'x' * 50} line {i}" for i in range(1200)])
        diff = _make_diff(_make_file("big.py", patch=large_patch))
        decisions = select_files(diff, max_diff_tokens_per_file=4000)
        assert len(decisions) == 1
        assert decisions[0].reason == ExcludeReason.TOO_LARGE

    def test_mixed_files(self):
        """Test a realistic scenario with multiple file types."""
        diff = _make_diff(
            _make_file("src/auth/login.py", additions=20, deletions=5),
            _make_file("logo.png"),
            _make_file("package-lock.json"),
            _make_file("src/utils.py", additions=10, deletions=3),
            _make_file("old_module.py", is_deleted=True),
        )
        decisions = select_files(diff)
        assert len(decisions) == 5

        summary = selection_summary(decisions)
        assert summary.get("none", 0) == 2   # login.py and utils.py
        assert summary.get("binary", 0) == 1  # logo.png
        assert summary.get("skip_pattern", 0) == 1  # package-lock.json
        assert summary.get("deleted", 0) == 1  # old_module.py

    def test_order_preserved(self):
        """File decisions should be in the same order as input files."""
        diff = _make_diff(
            _make_file("a.py"),
            _make_file("b.py"),
            _make_file("c.py"),
        )
        decisions = select_files(diff)
        assert [d.file.path for d in decisions] == ["a.py", "b.py", "c.py"]

    def test_custom_skip_patterns(self):
        diff = _make_diff(_make_file("internal/generated_api.py"))
        decisions = select_files(diff, skip_patterns=["internal/*"])
        assert decisions[0].reason == ExcludeReason.SKIP_PATTERN
        assert decisions[0].matched_pattern == "internal/*"

    def test_no_extension_allowlist(self):
        """When allowed_extensions is None, all extensions pass."""
        diff = _make_diff(_make_file("data.xyz"))
        decisions = select_files(diff, allowed_extensions=None, skip_patterns=[])
        assert decisions[0].selected is True

    def test_custom_token_ceiling(self):
        patch = "\n".join([f"+line {i}" for i in range(100)])
        diff = _make_diff(_make_file("test.py", patch=patch))
        # With low ceiling, file should be excluded
        decisions = select_files(diff, max_diff_tokens_per_file=100)
        assert decisions[0].reason == ExcludeReason.TOO_LARGE

    def test_vendor_directory_excluded(self):
        diff = _make_diff(_make_file("vendor/github.com/pkg/errors/errors.go"))
        decisions = select_files(diff)
        assert decisions[0].reason == ExcludeReason.SKIP_PATTERN

    def test_node_modules_excluded(self):
        diff = _make_diff(_make_file("node_modules/react/index.js"))
        decisions = select_files(diff)
        assert decisions[0].reason == ExcludeReason.SKIP_PATTERN


# ---------------------------------------------------------------------------
# Tests for FileDecision properties
# ---------------------------------------------------------------------------

class TestFileDecision:
    def test_selected_property(self):
        d = FileDecision(
            file=_make_file("test.py"),
            reason=ExcludeReason.NONE,
            diff_tokens=100,
        )
        assert d.selected is True
        assert d.retained is True

    def test_excluded_not_selected(self):
        d = FileDecision(
            file=_make_file("test.py"),
            reason=ExcludeReason.BINARY,
            diff_tokens=0,
        )
        assert d.selected is False
        assert d.retained is False

    def test_deleted_retained_not_selected(self):
        d = FileDecision(
            file=_make_file("test.py", is_deleted=True),
            reason=ExcludeReason.DELETED,
            diff_tokens=50,
        )
        assert d.selected is False
        assert d.retained is True

    def test_frozen_dataclass(self):
        d = FileDecision(
            file=_make_file("test.py"),
            reason=ExcludeReason.NONE,
            diff_tokens=100,
        )
        with pytest.raises(AttributeError):
            d.reason = ExcludeReason.BINARY  # type: ignore


# ---------------------------------------------------------------------------
# Tests for convenience helpers
# ---------------------------------------------------------------------------

class TestConvenienceHelpers:
    def test_selected_files(self):
        diff = _make_diff(
            _make_file("a.py"),
            _make_file("b.png"),
            _make_file("c.py"),
        )
        decisions = select_files(diff)
        result = selected_files(decisions)
        assert len(result) == 2
        assert all(f.path.endswith(".py") for f in result)

    def test_retained_files(self):
        diff = _make_diff(
            _make_file("a.py"),
            _make_file("old.py", is_deleted=True),
            _make_file("b.png"),
        )
        decisions = select_files(diff)
        result = retained_files(decisions)
        # a.py (selected) + old.py (deleted/retained)
        assert len(result) == 2

    def test_total_selected_tokens(self):
        diff = _make_diff(
            _make_file("a.py", patch="+line1\n+line2\n+line3"),
            _make_file("b.png"),  # binary, excluded
        )
        decisions = select_files(diff)
        tokens = total_selected_tokens(decisions)
        assert tokens > 0
        # Only a.py tokens should count
        assert tokens == _estimate_diff_tokens(_make_file("a.py", patch="+line1\n+line2\n+line3"))

    def test_selection_summary(self):
        diff = _make_diff(
            _make_file("a.py"),
            _make_file("b.png"),
            _make_file("internal/config.py"),
        )
        decisions = select_files(diff, skip_patterns=["internal/*"])
        summary = selection_summary(decisions)
        assert summary["none"] == 1
        assert summary["binary"] == 1
        assert summary["skip_pattern"] == 1

    def test_format_selection_preview(self):
        diff = _make_diff(
            _make_file("src/main.py"),
            _make_file("logo.png"),
        )
        decisions = select_files(diff)
        preview = format_selection_preview(decisions)
        assert "File Selection:" in preview
        assert "Selected: 1" in preview
        assert "Excluded: 1" in preview
        assert "src/main.py" in preview
        assert "logo.png" in preview


# ---------------------------------------------------------------------------
# Integration-style tests — filter priority / gate ordering
# ---------------------------------------------------------------------------

class TestGateOrdering:
    """Test that gates are applied in the documented order:
    binary → extension → skip_pattern → generated → deleted → too_large
    """

    def test_binary_takes_priority_over_skip_pattern(self):
        """A .png file should be excluded as BINARY, not SKIP_PATTERN,
        even though *.png is also in skip patterns."""
        diff = _make_diff(_make_file("image.png"))
        decisions = select_files(diff)
        # Binary gate runs first
        assert decisions[0].reason == ExcludeReason.BINARY

    def test_skip_pattern_takes_priority_over_generated(self):
        """A vendor file should be excluded as SKIP_PATTERN even if it
        has generated code markers."""
        patch = "// @generated\npackage vendor"
        diff = _make_diff(_make_file("vendor/lib.go", patch=patch))
        decisions = select_files(diff)
        assert decisions[0].reason == ExcludeReason.SKIP_PATTERN

    def test_deleted_takes_priority_over_too_large(self):
        """A deleted file should get DELETED reason, not TOO_LARGE."""
        large_patch = "\n".join([f"-line {i}" for i in range(2000)])
        diff = _make_diff(_make_file(
            "removed.py", is_deleted=True, patch=large_patch,
        ))
        decisions = select_files(diff, max_diff_tokens_per_file=100)
        assert decisions[0].reason == ExcludeReason.DELETED


# ---------------------------------------------------------------------------
# Tests for triage integration with file decisions
# ---------------------------------------------------------------------------

class TestTriageFromDecisions:
    """Test that triage_from_decisions works with select_files output."""

    def test_small_pr_fast_tier(self):
        from src.agents.triage import triage_from_decisions, ReviewTier

        diff = _make_diff(
            _make_file("src/utils.py", additions=5, deletions=2),
        )
        decisions = select_files(diff)
        tier, reason = triage_from_decisions(decisions)
        assert tier == ReviewTier.FAST

    def test_no_selected_files_fast_tier(self):
        from src.agents.triage import triage_from_decisions, ReviewTier

        diff = _make_diff(
            _make_file("logo.png"),
            _make_file("package-lock.json"),
        )
        decisions = select_files(diff)
        tier, reason = triage_from_decisions(decisions)
        assert tier == ReviewTier.FAST
        assert "No files selected" in reason

    def test_security_sensitive_deep_tier(self):
        from src.agents.triage import triage_from_decisions, ReviewTier

        # Security-sensitive file with >100 line changes
        diff = _make_diff(
            _make_file("src/auth/login.py", additions=80, deletions=30),
        )
        decisions = select_files(diff)
        tier, reason = triage_from_decisions(decisions)
        assert tier == ReviewTier.DEEP
        assert "Security" in reason

    def test_large_pr_deep_tier(self):
        from src.agents.triage import triage_from_decisions, ReviewTier

        # Many files → DEEP
        files = [_make_file(f"src/file_{i}.py", additions=50) for i in range(20)]
        diff = _make_diff(*files)
        decisions = select_files(diff)
        tier, reason = triage_from_decisions(decisions)
        assert tier == ReviewTier.DEEP
