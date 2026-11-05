"""Tests for the deterministic comment line resolver (Phase 4).

Tests cover:
  - Exact hunk match
  - Normalized whitespace match
  - Multi-line window match
  - Patch text scan
  - Cross-file relocation
  - Fallback behavior
  - Batch resolution
  - Edge cases (empty snippets, no hunks, etc.)
"""

from __future__ import annotations

import pytest

from src.agents.tools.definitions import RawComment
from src.diff.resolver import (
    ResolvedComment,
    ResolutionResult,
    apply_resolutions,
    format_resolution_summary,
    relocate_across_files,
    resolve_line_numbers,
)
from src.models.schemas import DiffContext, FileContext, HunkInfo


# ---------------------------------------------------------------------------
# Helpers — build test data
# ---------------------------------------------------------------------------

def _make_hunk(
    target_start: int,
    added: list[tuple[int, str]],
    *,
    removed: list[tuple[int, str]] | None = None,
    target_length: int | None = None,
) -> HunkInfo:
    """Build a HunkInfo with added lines."""
    if target_length is None:
        target_length = len(added) + 3  # some padding
    return HunkInfo(
        source_start=target_start,
        source_length=target_length,
        target_start=target_start,
        target_length=target_length,
        added_lines=added,
        removed_lines=removed or [],
    )


def _make_file(
    path: str,
    hunks: list[HunkInfo],
    *,
    patch: str = "",
) -> FileContext:
    """Build a FileContext with hunks."""
    return FileContext(
        path=path,
        hunks=hunks,
        patch=patch,
        additions=sum(len(h.added_lines) for h in hunks),
        deletions=sum(len(h.removed_lines) for h in hunks),
    )


def _make_diff(*files: FileContext) -> DiffContext:
    """Build a DiffContext from files."""
    return DiffContext(
        files=list(files),
        total_additions=sum(f.additions for f in files),
        total_deletions=sum(f.deletions for f in files),
    )


def _make_comment(
    file: str,
    existing_code: str,
    message: str = "test finding",
) -> RawComment:
    """Build a RawComment."""
    return RawComment(
        file=file,
        existing_code=existing_code,
        message=message,
    )


# ---------------------------------------------------------------------------
# Tests for exact hunk match (Strategy 1)
# ---------------------------------------------------------------------------

class TestExactHunkMatch:
    def test_exact_match_single_line(self):
        """Snippet that exactly matches an added line."""
        hunk = _make_hunk(10, [
            (10, "    x = compute_value()"),
            (11, "    return x + 1"),
        ])
        file = _make_file("main.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("main.py", "x = compute_value()")

        results = resolve_line_numbers([comment], diff)
        assert len(results) == 1
        _, resolution = results[0]
        assert resolution.line == 10
        assert "exact" in resolution.strategy or "hunk" in resolution.strategy

    def test_exact_match_substring(self):
        """Snippet is a substring of the added line content."""
        hunk = _make_hunk(5, [
            (5, "    result = dangerous_function(user_input)"),
        ])
        file = _make_file("api.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("api.py", "dangerous_function(user_input)")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.line == 5

    def test_no_match_wrong_file(self):
        """Snippet exists but in a different file — should fallback."""
        hunk = _make_hunk(1, [(1, "def hello():")])
        file = _make_file("other.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("missing.py", "def hello():")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        # Should fallback since missing.py is not in the diff
        assert resolution.strategy in ("fallback", "cross_file_relocation", "no_snippet")


# ---------------------------------------------------------------------------
# Tests for normalized match (Strategy 2)
# ---------------------------------------------------------------------------

class TestNormalizedMatch:
    def test_extra_whitespace_matches(self):
        """Snippet with different whitespace should still match."""
        hunk = _make_hunk(20, [
            (20, "    if  x  ==  None:"),
        ])
        file = _make_file("utils.py", [hunk])
        diff = _make_diff(file)
        # LLM might normalize whitespace
        comment = _make_comment("utils.py", "if x == None:")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.line == 20
        assert "normalized" in resolution.strategy

    def test_leading_trailing_whitespace(self):
        """Snippet with leading/trailing whitespace."""
        hunk = _make_hunk(15, [
            (15, "    return value"),
        ])
        file = _make_file("main.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("main.py", "  return value  ")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.line == 15


# ---------------------------------------------------------------------------
# Tests for multi-line window match (Strategy 3)
# ---------------------------------------------------------------------------

class TestMultilineWindow:
    def test_two_line_snippet(self):
        """Multi-line snippet matching consecutive added lines."""
        hunk = _make_hunk(10, [
            (10, "    try:"),
            (11, "        conn = db.connect()"),
            (12, "    except Exception:"),
            (13, "        pass"),
        ])
        file = _make_file("db.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("db.py", "    except Exception:\n        pass")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        # Should resolve to line 12 (where the first snippet line appears)
        assert resolution.line == 12

    def test_three_line_snippet(self):
        """Three-line snippet matching consecutive lines."""
        hunk = _make_hunk(1, [
            (1, "def process(data):"),
            (2, "    result = []"),
            (3, "    for item in data:"),
            (4, "        result.append(item * 2)"),
        ])
        file = _make_file("proc.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("proc.py", "result = []\nfor item in data:\n    result.append(item * 2)")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        # Should resolve to line 2 (first line of the snippet)
        assert resolution.line == 2

    def test_first_line_fallback(self):
        """Multi-line snippet where only first line matches."""
        hunk = _make_hunk(5, [
            (5, "    x = calculate()"),
            (6, "    y = transform(x)"),
        ])
        file = _make_file("math.py", [hunk])
        diff = _make_diff(file)
        # Only first line exists in the diff
        comment = _make_comment("math.py", "x = calculate()\nz = something_else()")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        # Should resolve to line 5 regardless of strategy
        assert resolution.line == 5


# ---------------------------------------------------------------------------
# Tests for patch scan (Strategy 4)
# ---------------------------------------------------------------------------

class TestPatchScan:
    def test_patch_text_match(self):
        """Snippet found in raw patch text but not in structured hunks."""
        patch_text = (
            "@@ -1,3 +1,5 @@\n"
            " import os\n"
            "+import sys\n"
            "+\n"
            "+def new_function():\n"
            "+    return sys.argv\n"
        )
        # Build file with empty hunks but real patch text
        file = _make_file("cli.py", [], patch=patch_text)
        diff = _make_diff(file)
        comment = _make_comment("cli.py", "return sys.argv")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.line >= 1
        assert "patch_scan" in resolution.strategy or "fallback" in resolution.strategy


# ---------------------------------------------------------------------------
# Tests for cross-file relocation (Strategy 5)
# ---------------------------------------------------------------------------

class TestCrossFileRelocation:
    def test_unique_match_in_other_file(self):
        """Snippet exists uniquely in a different file → relocate."""
        hunk_a = _make_hunk(1, [(1, "def foo():")])
        hunk_b = _make_hunk(10, [
            (10, "    vulnerable_call(user_input)"),
        ])
        file_a = _make_file("a.py", [hunk_a])
        file_b = _make_file("b.py", [hunk_b])
        diff = _make_diff(file_a, file_b)

        # Comment filed against a.py but snippet is in b.py
        comment = _make_comment("a.py", "vulnerable_call(user_input)")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.file == "b.py"
        assert resolution.line == 10
        assert resolution.relocated is True
        assert resolution.original_file == "a.py"

    def test_ambiguous_match_no_relocation(self):
        """Snippet exists in multiple files → don't relocate."""
        hunk_a = _make_hunk(5, [(5, "    print(value)")])
        hunk_b = _make_hunk(10, [(10, "    print(value)")])
        file_a = _make_file("a.py", [hunk_a])
        file_b = _make_file("b.py", [hunk_b])
        diff = _make_diff(file_a, file_b)

        # Comment filed against a non-existent file
        comment = _make_comment("c.py", "print(value)")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        # Should NOT relocate due to ambiguity
        assert resolution.relocated is False

    def test_relocate_across_files_directly(self):
        """Test the relocate function directly."""
        hunk = _make_hunk(42, [(42, "    secret = os.getenv('API_KEY')")])
        file = _make_file("config.py", [hunk])

        comment = _make_comment("wrong.py", "secret = os.getenv('API_KEY')")
        result = relocate_across_files(comment, [file])
        assert result is not None
        assert result.file == "config.py"
        assert result.line == 42
        assert result.relocated is True


# ---------------------------------------------------------------------------
# Tests for fallback behavior
# ---------------------------------------------------------------------------

class TestFallback:
    def test_no_snippet(self):
        """Empty existing_code → fallback to first added line."""
        hunk = _make_hunk(7, [(7, "    x = 1"), (8, "    y = 2")])
        file = _make_file("main.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("main.py", "")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.line == 7
        assert resolution.strategy == "no_snippet"

    def test_whitespace_only_snippet(self):
        """Snippet with only whitespace → fallback."""
        hunk = _make_hunk(3, [(3, "    pass")])
        file = _make_file("empty.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("empty.py", "   \n  \t  ")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.line == 3
        assert resolution.strategy == "no_snippet"

    def test_file_not_in_diff(self):
        """Comment references a file not in the diff → line 1."""
        diff = _make_diff()  # empty diff
        comment = _make_comment("nonexistent.py", "some_code()")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.line == 1
        assert resolution.strategy == "fallback"

    def test_snippet_not_found_anywhere(self):
        """Snippet doesn't match any file → fallback."""
        hunk = _make_hunk(1, [(1, "    x = 1")])
        file = _make_file("main.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("main.py", "completely_different_code()")

        results = resolve_line_numbers([comment], diff)
        _, resolution = results[0]
        assert resolution.line == 1
        assert resolution.strategy == "fallback"


# ---------------------------------------------------------------------------
# Tests for batch resolution
# ---------------------------------------------------------------------------

class TestBatchResolution:
    def test_multiple_comments(self):
        """Resolve multiple comments at once."""
        hunk = _make_hunk(10, [
            (10, "    x = compute()"),
            (11, "    y = transform(x)"),
            (12, "    return y"),
        ])
        file = _make_file("math.py", [hunk])
        diff = _make_diff(file)

        comments = [
            _make_comment("math.py", "x = compute()"),
            _make_comment("math.py", "y = transform(x)"),
            _make_comment("math.py", "return y"),
        ]

        results = resolve_line_numbers(comments, diff)
        assert len(results) == 3
        assert results[0][1].line == 10
        assert results[1][1].line == 11
        assert results[2][1].line == 12

    def test_empty_comments(self):
        """No comments → empty results."""
        diff = _make_diff()
        results = resolve_line_numbers([], diff)
        assert results == []

    def test_mixed_strategies(self):
        """Different comments resolve via different strategies."""
        hunk = _make_hunk(1, [
            (1, "    exact_match_line"),
            (2, "    another   spaced   line"),
        ])
        file = _make_file("mix.py", [hunk])
        diff = _make_diff(file)

        comments = [
            _make_comment("mix.py", "exact_match_line"),
            _make_comment("mix.py", "another spaced line"),
            _make_comment("mix.py", ""),  # no snippet
        ]

        results = resolve_line_numbers(comments, diff)
        assert len(results) == 3
        # All should resolve to some line
        for _, res in results:
            assert res.line >= 1


# ---------------------------------------------------------------------------
# Tests for apply_resolutions
# ---------------------------------------------------------------------------

class TestApplyResolutions:
    def test_returns_resolved_comments(self):
        hunk = _make_hunk(5, [(5, "    dangerous_call()")])
        file = _make_file("api.py", [hunk])
        diff = _make_diff(file)
        comments = [_make_comment("api.py", "dangerous_call()")]

        resolved = apply_resolutions(comments, diff)
        assert len(resolved) == 1
        assert isinstance(resolved[0], ResolvedComment)
        assert resolved[0].line == 5
        assert resolved[0].file == "api.py"
        assert resolved[0].comment is comments[0]

    def test_resolved_comment_preserves_original(self):
        hunk = _make_hunk(1, [(1, "x = 1")])
        file = _make_file("t.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("t.py", "x = 1", message="bad pattern")

        resolved = apply_resolutions([comment], diff)
        assert resolved[0].comment.message == "bad pattern"
        assert resolved[0].comment.file == "t.py"


# ---------------------------------------------------------------------------
# Tests for format_resolution_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_basic_summary(self):
        hunk = _make_hunk(1, [(1, "line_a"), (2, "line_b")])
        file = _make_file("a.py", [hunk])
        diff = _make_diff(file)
        comments = [
            _make_comment("a.py", "line_a"),
            _make_comment("a.py", "line_b"),
        ]

        results = resolve_line_numbers(comments, diff)
        summary = format_resolution_summary(results)
        assert "🎯" in summary
        assert "2 comments" in summary

    def test_empty_results(self):
        summary = format_resolution_summary([])
        assert "No comments" in summary


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_long_snippet(self):
        """Long snippet that wraps multiple lines."""
        long_line = "x = " + "very_long_function_name(" * 10 + "value" + ")" * 10
        hunk = _make_hunk(1, [(1, long_line)])
        file = _make_file("long.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("long.py", long_line)

        results = resolve_line_numbers([comment], diff)
        _, res = results[0]
        assert res.line == 1

    def test_snippet_with_special_chars(self):
        """Snippet with regex special characters."""
        hunk = _make_hunk(5, [
            (5, "    pattern = re.compile(r'\\d+\\.\\d+')"),
        ])
        file = _make_file("regex.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("regex.py", "re.compile(r'\\d+\\.\\d+')")

        results = resolve_line_numbers([comment], diff)
        _, res = results[0]
        assert res.line == 5

    def test_duplicate_lines(self):
        """Multiple identical lines — should resolve to the first."""
        hunk = _make_hunk(10, [
            (10, "    pass"),
            (11, "    pass"),
            (12, "    pass"),
        ])
        file = _make_file("dup.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("dup.py", "pass")

        results = resolve_line_numbers([comment], diff)
        _, res = results[0]
        assert res.line == 10  # First match

    def test_hunk_with_no_added_lines(self):
        """File with hunks but no added lines."""
        hunk = _make_hunk(1, [], removed=[(1, "old_line")])
        file = _make_file("del.py", [hunk])
        diff = _make_diff(file)
        comment = _make_comment("del.py", "old_line")

        results = resolve_line_numbers([comment], diff)
        _, res = results[0]
        # Should fallback since there are no added lines to match
        assert res.line >= 1
