"""Tests for the semantic file grouping module (Phase 5).

Tests cover:
  - Trivial grouping (0-1 files)
  - Small batch grouping (≤ threshold files)
  - Directory-based fallback grouping
  - LLM response parsing
  - Budget enforcement / group splitting
  - Full group_files() integration
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.file_selection import ExcludeReason, FileDecision
from src.agents.grouping import (
    DEFAULT_MAX_FILES_PER_GROUP,
    DEFAULT_TOKEN_LIMIT_PER_GROUP,
    FileGroup,
    SMALL_CHANGE_THRESHOLD,
    _build_grouping_prompt,
    _enforce_group_limits,
    _group_by_directory,
    _group_trivial,
    _infer_language,
    _parse_grouping_response,
    format_grouping_summary,
    group_files,
    total_group_tokens,
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
    is_deleted: bool = False,
    language: str | None = None,
) -> FileContext:
    """Create a minimal FileContext for testing."""
    lines = [f"+added line {i}" for i in range(additions)]
    lines += [f"-removed line {i}" for i in range(deletions)]
    patch = "\n".join(lines) if lines else ""
    return FileContext(
        path=path,
        language=language,
        is_deleted=is_deleted,
        patch=patch,
        additions=additions,
        deletions=deletions,
    )


def _make_decision(
    path: str,
    *,
    additions: int = 10,
    deletions: int = 5,
    diff_tokens: int = 60,
    selected: bool = True,
    language: str | None = None,
) -> FileDecision:
    """Create a FileDecision for testing."""
    file_ctx = _make_file(
        path,
        additions=additions,
        deletions=deletions,
        language=language,
    )
    reason = ExcludeReason.NONE if selected else ExcludeReason.BINARY
    return FileDecision(
        file=file_ctx,
        reason=reason,
        diff_tokens=diff_tokens,
    )


def _make_llm_response(content: str):
    """Create a mock LLM response object."""
    response = MagicMock()
    response.content = content
    response.tokens_used = 100
    return response


# ---------------------------------------------------------------------------
# Tests for _infer_language
# ---------------------------------------------------------------------------

class TestInferLanguage:
    def test_python(self):
        assert _infer_language("src/main.py") == "python"

    def test_typescript(self):
        assert _infer_language("app/index.tsx") == "typescript"

    def test_go(self):
        assert _infer_language("cmd/server.go") == "go"

    def test_config_yaml(self):
        assert _infer_language("config.yaml") == "config"

    def test_unknown_extension(self):
        assert _infer_language("data.xyz") == "unknown"

    def test_no_extension(self):
        assert _infer_language("Makefile") == "unknown"


# ---------------------------------------------------------------------------
# Tests for _group_trivial
# ---------------------------------------------------------------------------

class TestGroupTrivial:
    def test_empty_list(self):
        groups = _group_trivial([])
        assert groups == []

    def test_single_file(self):
        decisions = [_make_decision("src/main.py", diff_tokens=100)]
        groups = _group_trivial(decisions)
        assert len(groups) == 1
        assert groups[0].label == "All changes"
        assert groups[0].files == ["src/main.py"]
        assert groups[0].total_tokens == 100
        assert groups[0].grouping_method == "trivial"

    def test_multiple_files(self):
        decisions = [
            _make_decision("a.py", diff_tokens=50),
            _make_decision("b.py", diff_tokens=70),
        ]
        groups = _group_trivial(decisions)
        assert len(groups) == 1
        assert len(groups[0].files) == 2
        assert groups[0].total_tokens == 120
        assert groups[0].grouping_method == "small"


# ---------------------------------------------------------------------------
# Tests for _group_by_directory
# ---------------------------------------------------------------------------

class TestGroupByDirectory:
    def test_single_directory(self):
        decisions = [
            _make_decision("src/auth/login.py"),
            _make_decision("src/auth/session.py"),
        ]
        groups = _group_by_directory(decisions)
        assert len(groups) == 1
        assert groups[0].label == "Changes in src/"

    def test_multiple_directories(self):
        decisions = [
            _make_decision("src/auth/login.py"),
            _make_decision("tests/test_login.py"),
            _make_decision("config/settings.yaml"),
        ]
        groups = _group_by_directory(decisions)
        assert len(groups) == 3
        labels = {g.label for g in groups}
        assert "Changes in src/" in labels
        assert "Changes in tests/" in labels
        assert "Changes in config/" in labels

    def test_root_level_files(self):
        decisions = [
            _make_decision("README.md"),
            _make_decision("setup.py"),
        ]
        groups = _group_by_directory(decisions)
        assert len(groups) == 1
        assert groups[0].label == "Root-level changes"

    def test_mixed_root_and_nested(self):
        decisions = [
            _make_decision("README.md"),
            _make_decision("src/main.py"),
        ]
        groups = _group_by_directory(decisions)
        assert len(groups) == 2

    def test_grouping_method_is_fallback(self):
        decisions = [_make_decision("src/main.py")]
        groups = _group_by_directory(decisions)
        assert groups[0].grouping_method == "fallback"


# ---------------------------------------------------------------------------
# Tests for _parse_grouping_response
# ---------------------------------------------------------------------------

class TestParseGroupingResponse:
    def test_valid_json(self):
        decisions = [
            _make_decision("src/auth/login.py"),
            _make_decision("src/auth/session.py"),
            _make_decision("tests/test_login.py"),
        ]
        response = json.dumps({
            "groups": [
                {"label": "Auth module", "file_indices": [0, 1]},
                {"label": "Tests", "file_indices": [2]},
            ]
        })
        groups = _parse_grouping_response(response, decisions)
        assert groups is not None
        assert len(groups) == 2
        assert groups[0].label == "Auth module"
        assert groups[0].files == ["src/auth/login.py", "src/auth/session.py"]
        assert groups[1].label == "Tests"
        assert groups[1].files == ["tests/test_login.py"]

    def test_json_in_markdown_block(self):
        decisions = [
            _make_decision("a.py"),
            _make_decision("b.py"),
        ]
        response = '```json\n{"groups": [{"label": "All", "file_indices": [0, 1]}]}\n```'
        groups = _parse_grouping_response(response, decisions)
        assert groups is not None
        assert len(groups) == 1

    def test_invalid_json_returns_none(self):
        decisions = [_make_decision("a.py")]
        groups = _parse_grouping_response("not valid json", decisions)
        assert groups is None

    def test_missing_groups_key_returns_none(self):
        decisions = [_make_decision("a.py")]
        groups = _parse_grouping_response('{"data": []}', decisions)
        assert groups is None

    def test_out_of_range_indices_skipped(self):
        decisions = [_make_decision("a.py"), _make_decision("b.py")]
        response = json.dumps({
            "groups": [
                {"label": "Valid", "file_indices": [0, 999]},
            ]
        })
        groups = _parse_grouping_response(response, decisions)
        assert groups is not None
        # Index 999 is out of range, so only index 0 is included
        # b.py (index 1) goes to "Other changes" group
        assert len(groups[0].files) == 1
        assert groups[0].files[0] == "a.py"

    def test_duplicate_indices_deduplicated(self):
        decisions = [_make_decision("a.py"), _make_decision("b.py")]
        response = json.dumps({
            "groups": [
                {"label": "Group1", "file_indices": [0]},
                {"label": "Group2", "file_indices": [0, 1]},  # index 0 already used
            ]
        })
        groups = _parse_grouping_response(response, decisions)
        assert groups is not None
        # Group1 gets index 0, Group2 only gets index 1
        assert groups[0].files == ["a.py"]
        assert groups[1].files == ["b.py"]

    def test_unassigned_files_get_other_group(self):
        decisions = [
            _make_decision("a.py"),
            _make_decision("b.py"),
            _make_decision("c.py"),
        ]
        response = json.dumps({
            "groups": [
                {"label": "Partial", "file_indices": [0]},
            ]
        })
        groups = _parse_grouping_response(response, decisions)
        assert groups is not None
        assert len(groups) == 2
        assert groups[1].label == "Other changes"
        assert set(groups[1].files) == {"b.py", "c.py"}

    def test_empty_groups_returns_none(self):
        decisions = [_make_decision("a.py")]
        response = json.dumps({"groups": []})
        groups = _parse_grouping_response(response, decisions)
        assert groups is None


# ---------------------------------------------------------------------------
# Tests for _enforce_group_limits
# ---------------------------------------------------------------------------

class TestEnforceGroupLimits:
    def test_within_limits_no_change(self):
        groups = [
            FileGroup(id=0, label="OK", files=["a.py"], total_tokens=100,
                      file_decisions=[_make_decision("a.py")]),
        ]
        result = _enforce_group_limits(groups, max_files_per_group=10, token_limit=50_000)
        assert len(result) == 1
        assert result[0].label == "OK"

    def test_split_by_file_count(self):
        decisions = [_make_decision(f"file_{i}.py", diff_tokens=100) for i in range(15)]
        groups = [
            FileGroup(
                id=0, label="Big group",
                files=[d.file.path for d in decisions],
                file_decisions=decisions,
                total_tokens=1500,
            ),
        ]
        result = _enforce_group_limits(groups, max_files_per_group=5, token_limit=100_000)
        assert len(result) == 3  # 15 files / 5 per group = 3 groups
        # All files accounted for
        all_files = [f for g in result for f in g.files]
        assert len(all_files) == 15

    def test_split_by_token_limit(self):
        decisions = [
            _make_decision("a.py", diff_tokens=30_000),
            _make_decision("b.py", diff_tokens=30_000),
        ]
        groups = [
            FileGroup(
                id=0, label="Heavy",
                files=["a.py", "b.py"],
                file_decisions=decisions,
                total_tokens=60_000,
            ),
        ]
        result = _enforce_group_limits(groups, max_files_per_group=10, token_limit=40_000)
        assert len(result) == 2
        # Each file in its own group since 30k + 30k > 40k limit
        assert len(result[0].files) == 1
        assert len(result[1].files) == 1

    def test_split_preserves_method(self):
        decisions = [_make_decision(f"f_{i}.py", diff_tokens=100) for i in range(6)]
        groups = [
            FileGroup(
                id=0, label="LLM group",
                files=[d.file.path for d in decisions],
                file_decisions=decisions,
                total_tokens=600,
                grouping_method="llm_semantic",
            ),
        ]
        result = _enforce_group_limits(groups, max_files_per_group=3, token_limit=100_000)
        assert len(result) == 2
        assert all(g.grouping_method == "llm_semantic" for g in result)

    def test_split_labels_include_part_number(self):
        decisions = [_make_decision(f"f_{i}.py") for i in range(8)]
        groups = [
            FileGroup(
                id=0, label="Big group",
                files=[d.file.path for d in decisions],
                file_decisions=decisions,
                total_tokens=480,
            ),
        ]
        result = _enforce_group_limits(groups, max_files_per_group=3, token_limit=100_000)
        assert len(result) == 3
        assert "part 1" in result[0].label
        assert "part 2" in result[1].label
        assert "part 3" in result[2].label

    def test_ids_are_sequential(self):
        decisions = [_make_decision(f"f_{i}.py") for i in range(6)]
        groups = [
            FileGroup(id=0, label="G1", files=["f_0.py", "f_1.py", "f_2.py"],
                      file_decisions=decisions[:3], total_tokens=180),
            FileGroup(id=1, label="G2", files=["f_3.py", "f_4.py", "f_5.py"],
                      file_decisions=decisions[3:], total_tokens=180),
        ]
        result = _enforce_group_limits(groups, max_files_per_group=2, token_limit=100_000)
        ids = [g.id for g in result]
        assert ids == list(range(len(result)))


# ---------------------------------------------------------------------------
# Tests for group_files — main entry point
# ---------------------------------------------------------------------------

class TestGroupFiles:
    @pytest.mark.asyncio
    async def test_empty_decisions(self):
        groups = await group_files([])
        assert groups == []

    @pytest.mark.asyncio
    async def test_excluded_files_ignored(self):
        decisions = [
            _make_decision("a.py", selected=False),
            _make_decision("b.png", selected=False),
        ]
        groups = await group_files(decisions)
        assert groups == []

    @pytest.mark.asyncio
    async def test_single_file_trivial(self):
        decisions = [_make_decision("src/main.py", diff_tokens=100)]
        groups = await group_files(decisions)
        assert len(groups) == 1
        assert groups[0].grouping_method == "trivial"
        assert groups[0].files == ["src/main.py"]

    @pytest.mark.asyncio
    async def test_small_batch_no_llm(self):
        """≤ SMALL_CHANGE_THRESHOLD files should be grouped without LLM."""
        decisions = [
            _make_decision(f"src/file_{i}.py", diff_tokens=100)
            for i in range(SMALL_CHANGE_THRESHOLD)
        ]
        # Pass a mock LLM — it should NOT be called
        mock_llm = AsyncMock()
        groups = await group_files(decisions, llm=mock_llm)
        assert len(groups) >= 1
        mock_llm.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_batch_with_llm(self):
        """More than threshold files should use LLM grouping."""
        decisions = [
            _make_decision(f"src/file_{i}.py", diff_tokens=100)
            for i in range(SMALL_CHANGE_THRESHOLD + 3)
        ]
        llm_response = _make_llm_response(json.dumps({
            "groups": [
                {"label": "Core", "file_indices": list(range(4))},
                {"label": "Utils", "file_indices": list(range(4, len(decisions)))},
            ]
        }))
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = llm_response

        groups = await group_files(decisions, llm=mock_llm)
        assert len(groups) == 2
        assert groups[0].label == "Core"
        assert groups[0].grouping_method == "llm_semantic"
        mock_llm.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_directory(self):
        """When LLM fails, fall back to directory grouping."""
        decisions = [
            _make_decision("src/auth/login.py"),
            _make_decision("src/auth/session.py"),
            _make_decision("tests/test_login.py"),
            _make_decision("tests/test_session.py"),
            _make_decision("config/settings.yaml"),
            _make_decision("docs/readme.md"),
        ]
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = None  # LLM failure

        groups = await group_files(decisions, llm=mock_llm)
        assert len(groups) >= 2
        assert all(g.grouping_method == "fallback" for g in groups)

    @pytest.mark.asyncio
    async def test_no_llm_uses_directory_fallback(self):
        """When llm is None, use directory grouping for large sets."""
        decisions = [
            _make_decision(f"src/file_{i}.py")
            for i in range(SMALL_CHANGE_THRESHOLD + 1)
        ]
        groups = await group_files(decisions, llm=None)
        assert len(groups) >= 1
        assert all(g.grouping_method == "fallback" for g in groups)

    @pytest.mark.asyncio
    async def test_budget_enforcement_applied(self):
        """Groups exceeding token limits should be split."""
        decisions = [
            _make_decision(f"file_{i}.py", diff_tokens=20_000)
            for i in range(3)
        ]
        groups = await group_files(
            decisions,
            llm=None,
            token_limit=30_000,  # 3 files × 20k = 60k > 30k
        )
        # Should be split into at least 2 groups
        total_files = sum(len(g.files) for g in groups)
        assert total_files == 3
        assert all(g.total_tokens <= 30_000 for g in groups)


# ---------------------------------------------------------------------------
# Tests for _build_grouping_prompt
# ---------------------------------------------------------------------------

class TestBuildGroupingPrompt:
    def test_contains_file_info(self):
        decisions = [
            _make_decision("src/main.py", language="python", diff_tokens=200),
            _make_decision("tests/test_main.py", language="python", diff_tokens=100),
        ]
        prompt = _build_grouping_prompt(decisions)
        assert "src/main.py" in prompt
        assert "tests/test_main.py" in prompt
        assert "tokens≈200" in prompt
        assert "tokens≈100" in prompt

    def test_contains_json_structure_hint(self):
        decisions = [_make_decision("a.py")]
        prompt = _build_grouping_prompt(decisions)
        assert "file_indices" in prompt
        assert "JSON" in prompt

    def test_includes_max_files_rule(self):
        decisions = [_make_decision("a.py")]
        prompt = _build_grouping_prompt(decisions)
        assert str(DEFAULT_MAX_FILES_PER_GROUP) in prompt


# ---------------------------------------------------------------------------
# Tests for convenience helpers
# ---------------------------------------------------------------------------

class TestConvenienceHelpers:
    def test_format_grouping_summary(self):
        groups = [
            FileGroup(id=0, label="Auth", files=["login.py", "session.py"],
                      total_tokens=500, grouping_method="llm_semantic"),
            FileGroup(id=1, label="Tests", files=["test_login.py"],
                      total_tokens=200, grouping_method="llm_semantic"),
        ]
        summary = format_grouping_summary(groups)
        assert "2 groups" in summary
        assert "3 files total" in summary
        assert "Auth" in summary
        assert "Tests" in summary
        assert "llm_semantic" in summary

    def test_format_grouping_summary_truncates_files(self):
        files = [f"file_{i}.py" for i in range(8)]
        groups = [
            FileGroup(id=0, label="Big", files=files, total_tokens=800),
        ]
        summary = format_grouping_summary(groups)
        assert "and 3 more" in summary

    def test_total_group_tokens(self):
        groups = [
            FileGroup(id=0, label="A", total_tokens=500),
            FileGroup(id=1, label="B", total_tokens=300),
        ]
        assert total_group_tokens(groups) == 800
