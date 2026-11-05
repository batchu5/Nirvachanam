"""Tests for the MCP server tools (LLM and GitHub mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src import mcp_server
from src.models.schemas import Finding
from tests.conftest import SAMPLE_DIFF, SAMPLE_DIFF_WITH_SECRETS


async def test_all_tools_registered():
    names = {t.name for t in await mcp_server.mcp.list_tools()}
    assert names == {"preview_review", "scan_secrets", "review_diff", "review_pull_request"}


async def test_preview_review_is_zero_token_plan():
    with patch.object(mcp_server, "get_llm") as get_llm:
        plan = await mcp_server.preview_review_impl(SAMPLE_DIFF)
    get_llm.assert_not_called()
    assert plan["selected_files"] == ["src/auth/login.py"]
    assert plan["tier"] in {"fast", "standard", "deep"}
    assert "tokens" in plan["estimate"]


async def test_preview_review_reports_skipped_lockfile():
    diff = SAMPLE_DIFF + (
        "diff --git a/web/package-lock.json b/web/package-lock.json\n"
        "--- a/web/package-lock.json\n+++ b/web/package-lock.json\n"
        "@@ -1,1 +1,1 @@\n-a\n+b\n"
    )
    plan = await mcp_server.preview_review_impl(diff)
    assert "web/package-lock.json" not in plan["selected_files"]


async def test_scan_secrets_never_echoes_values():
    result = await mcp_server.scan_secrets_impl(SAMPLE_DIFF_WITH_SECRETS)
    assert result["secrets_found"] >= 2
    assert "AKIAIOSFODNN7EXAMPLE" not in str(result)


def _finding() -> Finding:
    return Finding(
        file="src/auth/login.py", line=14, severity="critical", category="security",
        message="SQL injection", confidence=0.9, agent="reviewer",
    )


_PIPELINE_RESULT = {
    "summary": "1 issue",
    "verified_findings": [_finding()],
    "failed_agents": [],
    "degraded_mode": False,
    "review_tier": "standard",
}


async def test_review_diff_returns_serialisable_findings():
    with (
        patch.object(mcp_server, "get_llm", return_value=object()),
        patch.object(mcp_server, "run_review_pipeline", AsyncMock(return_value=_PIPELINE_RESULT)),
    ):
        out = await mcp_server.review_diff_impl(SAMPLE_DIFF)
    assert out["findings"][0]["severity"] == "critical"
    assert out["findings"][0]["file"] == "src/auth/login.py"


@pytest.fixture
def github():
    gh = AsyncMock()
    gh.get_pr.return_value = {
        "title": "t", "user": {"login": "u"},
        "head": {"sha": "a" * 40, "ref": "f"}, "base": {"sha": "b" * 40, "ref": "main"},
    }
    gh.get_pr_diff.return_value = SAMPLE_DIFF
    return gh


async def test_review_pull_request_does_not_post_by_default(github):
    with (
        patch.object(mcp_server, "get_llm", return_value=object()),
        patch.object(mcp_server, "run_review_pipeline", AsyncMock(return_value=_PIPELINE_RESULT)),
    ):
        out = await mcp_server.review_pull_request_impl("o/r", 1, github=github)
    assert len(out["findings"]) == 1
    github.post_review.assert_not_awaited()


async def test_review_pull_request_post_requires_installation(github):
    with (
        patch.object(mcp_server, "get_llm", return_value=object()),
        patch.object(mcp_server, "run_review_pipeline", AsyncMock(return_value=_PIPELINE_RESULT)),
    ):
        out = await mcp_server.review_pull_request_impl("o/r", 1, post_review=True, github=github)
    assert out["posted"] is False
    github.post_review.assert_not_awaited()


async def test_review_pull_request_posts_inline_review(github):
    with (
        patch.object(mcp_server, "get_llm", return_value=object()),
        patch.object(mcp_server, "run_review_pipeline", AsyncMock(return_value=_PIPELINE_RESULT)),
    ):
        out = await mcp_server.review_pull_request_impl(
            "o/r", 1, post_review=True, installation_id=5, github=github
        )
    assert out["posted"] is True
    args, kwargs = github.post_review.call_args
    assert args[2] == "a" * 40 and kwargs["installation_id"] == 5
    assert args[4][0]["path"] == "src/auth/login.py"


async def test_in_memory_quota_skips_exhausted_provider():
    q = mcp_server.InMemoryQuotaTracker()
    assert await q.can_call("gemini")
    await q.record_exhaustion("gemini", retry_after=60)
    assert not await q.can_call("gemini")
    assert await q.can_call("groq")
