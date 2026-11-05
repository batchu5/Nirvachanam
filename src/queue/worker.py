"""arq worker — processes review jobs from the Redis queue.

This is the worker process entry point. When a debounced job fires it runs
the full review end to end:

    staleness check → fetch PR + diff → redact + parse → LangGraph pipeline
    → map findings to diff positions → post one GitHub review

Run with:
    arq src.queue.worker.WorkerSettings

References: PRD §5 (flow), §11 (retries).
"""

from __future__ import annotations

from typing import Any

import structlog

from src.config import settings
from src.diff.parser import DiffParseError, build_diff_context
from src.github.client import GitHubClient
from src.github.context import current_github_client
from src.github.reviewer import build_review_body, build_review_comments
from src.graph.pipeline import run_review_pipeline
from src.llm.fallback import AllProvidersExhaustedError, QuotaAwareFallbackLLM
from src.llm.providers import GeminiProvider, GroqProvider
from src.llm.quota import ProviderQuotaTracker
from src.models.schemas import PRMetadata
from src.queue.debounce import check_already_reviewed, mark_as_reviewed
from src.queue.redis_pool import get_redis_settings

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_llm(redis: Any) -> QuotaAwareFallbackLLM:
    """Build the fallback LLM from whichever provider keys are configured.

    Order matters: the first provider is preferred, later ones are used when
    it is rate-limited or out of quota.
    """
    providers = []
    if settings.gemini_api_key:
        providers.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model))
    if settings.groq_api_key:
        providers.append(GroqProvider(settings.groq_api_key, settings.groq_model))
    if not providers:
        raise RuntimeError("No LLM provider configured: set GEMINI_API_KEY and/or GROQ_API_KEY")
    return QuotaAwareFallbackLLM(providers=providers, quota=ProviderQuotaTracker(redis))


def pr_metadata_from_api(repo: str, pr: dict[str, Any], installation_id: int | None) -> PRMetadata:
    """Build PRMetadata from a GitHub ``GET /pulls/{n}`` response."""
    return PRMetadata(
        repo=repo,
        pr_number=pr["number"],
        title=pr.get("title", ""),
        author=(pr.get("user") or {}).get("login", ""),
        base_branch=pr["base"].get("ref", ""),
        head_branch=pr["head"].get("ref", ""),
        installation_id=installation_id,
        head_sha=pr["head"]["sha"],
        base_sha=pr["base"]["sha"],
    )


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

async def run_review(
    ctx: dict,
    repo: str,
    pr: int,
    sha: str,
    installation_id: int | None = None,
) -> dict:
    """Review one PR at one commit and post the result to GitHub.

    Args:
        ctx: arq worker context (holds ``redis``, ``github`` and ``checkpointer``).
        repo: Repository full name (e.g. "owner/repo").
        pr: PR number.
        sha: HEAD commit SHA the webhook was for.
        installation_id: GitHub App installation ID (from the webhook).

    Returns:
        Dict with job result summary.
    """
    redis = ctx["redis"]
    github: GitHubClient = ctx["github"]
    log = logger.bind(repo=repo, pr=pr, sha=sha[:8])
    log.info("worker.run_review.start")

    # 1. Idempotency — this exact commit was already reviewed
    if await check_already_reviewed(redis, repo, pr, sha):
        log.info("worker.run_review.already_reviewed")
        return {"status": "skipped", "reason": "already_reviewed"}

    # 2. Staleness — a newer push will get its own job, don't review old code
    pr_data = await github.get_pr(repo, pr, installation_id=installation_id)
    current_head = pr_data["head"]["sha"]
    if current_head != sha:
        log.info("worker.run_review.stale", current_head=current_head[:8])
        return {"status": "skipped", "reason": "stale_sha"}
    if pr_data.get("state") != "open":
        log.info("worker.run_review.not_open", state=pr_data.get("state"))
        return {"status": "skipped", "reason": "pr_not_open"}

    metadata = pr_metadata_from_api(repo, pr_data, installation_id)

    # 3. Fetch diff → redact secrets → parse
    raw_diff = await github.get_pr_diff(repo, pr, installation_id=installation_id)
    try:
        diff_context = build_diff_context(raw_diff, metadata)
    except DiffParseError as exc:
        log.error("worker.run_review.diff_unparseable", error=str(exc))
        await github.post_comment(
            repo,
            pr,
            "⚠️ **AI review skipped:** the diff for this commit could not be parsed. "
            "Push another commit to retry.",
            installation_id=installation_id,
        )
        await mark_as_reviewed(redis, repo, pr, sha)
        return {"status": "failed", "reason": "diff_unparseable"}

    if not diff_context.files:
        log.info("worker.run_review.nothing_to_review")
        await mark_as_reviewed(redis, repo, pr, sha)
        return {"status": "skipped", "reason": "no_reviewable_files"}

    # 4. Run the LangGraph pipeline, giving the agent tools repo access
    token = current_github_client.set(github)
    try:
        result = await run_review_pipeline(
            diff_context,
            metadata,
            ctx["llm"],
            checkpointer=ctx.get("checkpointer"),
        )
    except AllProvidersExhaustedError:
        # Let arq retry later — quota windows reset.
        log.warning("worker.run_review.llm_quota_exhausted")
        raise
    finally:
        current_github_client.reset(token)

    # 5. Post a single review with inline comments
    findings = result["verified_findings"][: settings.max_comments_per_review]
    comments = build_review_comments(findings, diff_context)
    body = build_review_body(
        result["summary"],
        result["verified_findings"],
        degraded_agents=result["failed_agents"] or None,
    )
    await github.post_review(
        repo,
        pr,
        sha,
        body,
        comments,
        installation_id=installation_id,
    )

    await mark_as_reviewed(redis, repo, pr, sha)
    log.info(
        "worker.run_review.completed",
        findings=len(result["verified_findings"]),
        inline_comments=len(comments),
        tier=result.get("review_tier"),
    )
    return {
        "status": "completed",
        "repo": repo,
        "pr": pr,
        "sha": sha,
        "findings": len(result["verified_findings"]),
        "inline_comments": len(comments),
    }


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def startup(ctx: dict) -> None:
    """Create shared clients once per worker process."""
    from src.graph.checkpointer import create_postgres_checkpointer

    ctx["github"] = GitHubClient()
    ctx["llm"] = build_llm(ctx["redis"])
    # Optional: resume interrupted runs from Postgres. Reviews still work without it.
    ctx["checkpointer"] = await create_postgres_checkpointer()
    logger.info("worker.startup", checkpointer=ctx["checkpointer"] is not None)


async def shutdown(ctx: dict) -> None:
    """Close shared clients."""
    from src.graph.checkpointer import close_checkpointer_pool

    github = ctx.get("github")
    if github is not None:
        await github.close()
    await close_checkpointer_pool()
    logger.info("worker.shutdown")


class WorkerSettings:
    """arq worker configuration.

    This class is discovered by arq's CLI:
        arq src.queue.worker.WorkerSettings
    """

    functions = [run_review]
    on_startup = startup
    on_shutdown = shutdown

    redis_settings = get_redis_settings()

    # Worker behavior
    max_jobs = 5            # Max concurrent review jobs
    job_timeout = 600       # Agent loops on large PRs can take several minutes
    max_tries = 3           # Retry failed jobs up to 3 times
    health_check_interval = 30

    # Allow aborting deferred jobs (needed for debounce cancellation)
    allow_abort_jobs = True
