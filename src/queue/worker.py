"""arq worker — processes review jobs from the Redis queue.

This is the worker process entry point. It defines the job functions that arq
will execute when deferred jobs fire, and the WorkerSettings class that
configures the worker.

For M1 (Foundation), the worker is a stub — it logs the parsed diff context
but doesn't call any LLM. The full LangGraph pipeline integration comes in M2.

References: PRD §5 (flow), §11 (retries).
"""

from __future__ import annotations

import structlog
from arq.connections import RedisSettings

from src.config import settings
from src.queue.debounce import check_already_reviewed, mark_as_reviewed
from src.queue.redis_pool import get_redis_settings

logger = structlog.get_logger()


async def run_review(ctx: dict, repo: str, pr: int, sha: str) -> dict:
    """Execute a PR review — triggered by arq after debounce window elapses.

    M1 stub: parses the diff and logs context. Does NOT call LLMs or
    post to GitHub (that's M2+).

    Flow:
      1. Idempotency check — skip if this SHA was already reviewed
      2. (M2+) Staleness check — skip if PR HEAD moved past our SHA
      3. Log diff context summary
      4. Mark as reviewed

    Args:
        ctx: arq worker context (contains redis pool).
        repo: Repository full name (e.g. "owner/repo").
        pr: PR number.
        sha: HEAD commit SHA to review.

    Returns:
        Dict with job result summary.
    """
    redis = ctx["redis"]

    logger.info(
        "worker.run_review.start",
        repo=repo,
        pr=pr,
        sha=sha,
    )

    # Step 1: Idempotency check
    if await check_already_reviewed(redis, repo, pr, sha):
        logger.info(
            "worker.run_review.already_reviewed",
            repo=repo,
            pr=pr,
            sha=sha,
        )
        return {"status": "skipped", "reason": "already_reviewed"}

    # Step 2: (M2+) Staleness check against PR HEAD
    # TODO: Implement GitHub API call to get current PR HEAD SHA
    # current_head = await github.get_pr_head_sha(repo, pr)
    # if current_head != sha:
    #     logger.info("worker.run_review.stale", ...)
    #     return {"status": "skipped", "reason": "stale_sha"}

    # Step 3: (M1 stub) Log that we WOULD execute the pipeline
    logger.info(
        "worker.run_review.executing",
        repo=repo,
        pr=pr,
        sha=sha,
        message="M1 stub — would execute LangGraph pipeline here",
    )

    # TODO (M2): Fetch diff from GitHub API
    # TODO (M2): Build DiffContext via build_diff_context()
    # TODO (M2): Execute LangGraph pipeline
    # TODO (M2): Post review via GitHub Review API

    # Step 4: Mark as reviewed (idempotency)
    await mark_as_reviewed(redis, repo, pr, sha)

    logger.info(
        "worker.run_review.completed",
        repo=repo,
        pr=pr,
        sha=sha,
    )

    return {"status": "completed", "repo": repo, "pr": pr, "sha": sha}


async def startup(ctx: dict) -> None:
    """arq worker startup hook — runs once when the worker process starts."""
    logger.info("worker.startup")
    # TODO (M2): Initialize database connection pool
    # TODO (M2): Initialize GitHub API client


async def shutdown(ctx: dict) -> None:
    """arq worker shutdown hook — runs once when the worker stops."""
    logger.info("worker.shutdown")
    # TODO (M2): Close database connections


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
    job_timeout = 300       # 5 minutes max per review
    max_tries = 3           # Retry failed jobs up to 3 times
    health_check_interval = 30

    # Allow aborting deferred jobs (needed for debounce cancellation)
    allow_abort_jobs = True
