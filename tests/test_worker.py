"""Tests for the arq review job — GitHub and the LLM pipeline are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.github.context import current_github_client
from src.models.schemas import Finding
from src.queue import worker
from tests.conftest import SAMPLE_DIFF

REPO = "acme/api"
PR = 7
SHA = "a" * 40


def _pr_response(head_sha: str = SHA, state: str = "open") -> dict:
    return {
        "number": PR,
        "state": state,
        "title": "Add login",
        "user": {"login": "dev"},
        "head": {"sha": head_sha, "ref": "feature"},
        "base": {"sha": "b" * 40, "ref": "main"},
    }


@pytest.fixture
def github():
    gh = AsyncMock()
    gh.get_pr.return_value = _pr_response()
    gh.get_pr_diff.return_value = SAMPLE_DIFF
    return gh


@pytest.fixture
def ctx(github):
    return {"redis": AsyncMock(), "github": github, "llm": object(), "checkpointer": None}


def _finding(line: int) -> Finding:
    return Finding(
        file="src/auth/login.py",
        line=line,
        severity="critical",
        category="security",
        message="SQL injection via f-string query",
        confidence=0.9,
        agent="reviewer",
    )


@pytest.fixture(autouse=True)
def not_reviewed_yet():
    with (
        patch.object(worker, "check_already_reviewed", AsyncMock(return_value=False)),
        patch.object(worker, "mark_as_reviewed", AsyncMock()) as mark,
    ):
        yield mark


async def test_happy_path_posts_review_with_inline_comment(ctx, github, not_reviewed_yet):
    seen_client = {}

    async def fake_pipeline(diff_context, metadata, llm, checkpointer=None):
        seen_client["value"] = current_github_client.get()
        assert metadata.head_sha == SHA
        assert [f.path for f in diff_context.files] == ["src/auth/login.py"]
        return {
            "summary": "Found 1 issue.",
            "verified_findings": [_finding(14)],
            "failed_agents": [],
            "review_tier": "standard",
        }

    with patch.object(worker, "run_review_pipeline", fake_pipeline):
        result = await worker.run_review(ctx, REPO, PR, SHA, installation_id=99)

    assert result["status"] == "completed"
    assert result["inline_comments"] == 1
    # Agent tools could reach the repo during the run, and the var is reset after.
    assert seen_client["value"] is github
    assert current_github_client.get() is None

    github.post_review.assert_awaited_once()
    args, kwargs = github.post_review.call_args
    assert args[:3] == (REPO, PR, SHA)
    assert kwargs["installation_id"] == 99
    comments = args[4]
    assert comments[0]["path"] == "src/auth/login.py"
    not_reviewed_yet.assert_awaited_once()


async def test_stale_sha_is_skipped(ctx, github):
    github.get_pr.return_value = _pr_response(head_sha="c" * 40)
    result = await worker.run_review(ctx, REPO, PR, SHA)
    assert result == {"status": "skipped", "reason": "stale_sha"}
    github.get_pr_diff.assert_not_awaited()
    github.post_review.assert_not_awaited()


async def test_closed_pr_is_skipped(ctx, github):
    github.get_pr.return_value = _pr_response(state="closed")
    result = await worker.run_review(ctx, REPO, PR, SHA)
    assert result["reason"] == "pr_not_open"


async def test_already_reviewed_is_skipped(ctx, github):
    with patch.object(worker, "check_already_reviewed", AsyncMock(return_value=True)):
        result = await worker.run_review(ctx, REPO, PR, SHA)
    assert result["reason"] == "already_reviewed"
    github.get_pr.assert_not_awaited()


async def test_unparseable_diff_posts_comment_instead_of_silence(ctx, github):
    github.get_pr_diff.return_value = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,5 +1,6 @@\n+x\n"
    )
    result = await worker.run_review(ctx, REPO, PR, SHA)
    assert result["reason"] == "diff_unparseable"
    github.post_comment.assert_awaited_once()
    github.post_review.assert_not_awaited()


async def test_only_lockfiles_means_nothing_to_review(ctx, github):
    github.get_pr_diff.return_value = (
        "diff --git a/web/package-lock.json b/web/package-lock.json\n"
        "--- a/web/package-lock.json\n+++ b/web/package-lock.json\n"
        "@@ -1,1 +1,1 @@\n-a\n+b\n"
    )
    with patch.object(worker, "run_review_pipeline", AsyncMock()) as pipeline:
        result = await worker.run_review(ctx, REPO, PR, SHA)
    assert result["reason"] == "no_reviewable_files"
    pipeline.assert_not_awaited()


def test_build_llm_requires_a_provider(monkeypatch):
    monkeypatch.setattr(worker.settings, "gemini_api_key", "")
    monkeypatch.setattr(worker.settings, "groq_api_key", "")
    with pytest.raises(RuntimeError, match="No LLM provider"):
        worker.build_llm(AsyncMock())


_FAILED_RESULT = {
    "summary": "x",
    "verified_findings": [],
    "failed_agents": ["reviewer"],
    "review_tier": "standard",
}


async def test_llm_outage_retries_instead_of_posting_clean_review(ctx, github, not_reviewed_yet):
    from arq import Retry

    ctx["job_try"] = 1
    with patch.object(worker, "run_review_pipeline", AsyncMock(return_value=_FAILED_RESULT)):
        with pytest.raises(Retry):
            await worker.run_review(ctx, REPO, PR, SHA)
    github.post_review.assert_not_awaited()
    not_reviewed_yet.assert_not_awaited()


async def test_llm_outage_on_last_try_posts_visible_failure(ctx, github, not_reviewed_yet):
    ctx["job_try"] = worker.WorkerSettings.max_tries
    with patch.object(worker, "run_review_pipeline", AsyncMock(return_value=_FAILED_RESULT)):
        result = await worker.run_review(ctx, REPO, PR, SHA)
    assert result["reason"] == "llm_unavailable"
    github.post_review.assert_not_awaited()
    assert "could not be completed" in github.post_comment.call_args.args[2]
