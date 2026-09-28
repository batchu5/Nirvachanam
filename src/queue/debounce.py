"""Debounce logic for PR review jobs using arq deferred scheduling.

When multiple pushes arrive in rapid succession for the same PR, we want to
review only the LAST push, not every intermediate one. This is implemented
by cancelling any pending deferred job and re-enqueuing with a fresh delay.

Flow (PRD §5):
  1. Each push → cancel existing deferred job for this PR (if any)
  2. Enqueue new `run_review` job deferred by REVIEW_DEBOUNCE_SECONDS
  3. After the window elapses, arq fires the job → worker picks it up
  4. If another push arrived during the window, step 1 already cancelled us

Note: arq's job_id dedup is per (function_name, job_id). We use
`review:{repo}:{pr}` as the job_id for natural dedup.
"""

from __future__ import annotations

import os
from datetime import timedelta

import structlog
from arq.connections import ArqRedis

from src.config import settings

logger = structlog.get_logger()


async def handle_push_webhook(
    redis: ArqRedis,
    repo: str,
    pr: int,
    sha: str,
) -> None:
    """Enqueue a debounced review job for a PR push event.

    On each push: enqueue a deferred job with a fixed job_id per (repo, pr).
    If a push arrives while a deferred job is pending, abort the old job
    and re-enqueue with a fresh deferral — effectively resetting the
    debounce window.

    Args:
        redis: arq Redis connection pool.
        repo: Repository full name (e.g. "owner/repo").
        pr: PR number.
        sha: HEAD commit SHA of the push.
    """
    job_id = f"review:{repo}:{pr}"
    debounce_seconds = settings.review_debounce_seconds

    # Cancel any pending deferred job for this PR
    try:
        existing_jobs = await redis.queued_jobs()
        for job in existing_jobs:
            if job.job_id == job_id:
                await job.abort()
                logger.info(
                    "debounce.cancelled_previous",
                    job_id=job_id,
                    repo=repo,
                    pr=pr,
                )
                break
    except Exception:
        # If queued_jobs fails (e.g. no jobs), just proceed — not critical
        logger.debug("debounce.no_existing_jobs", job_id=job_id)

    # Enqueue new job deferred by debounce window
    await redis.enqueue_job(
        "run_review",
        repo=repo,
        pr=pr,
        sha=sha,
        _job_id=job_id,
        _defer_by=timedelta(seconds=debounce_seconds),
    )

    logger.info(
        "debounce.enqueued",
        job_id=job_id,
        repo=repo,
        pr=pr,
        sha=sha,
        defer_seconds=debounce_seconds,
    )


async def check_already_reviewed(redis: ArqRedis, repo: str, pr: int, sha: str) -> bool:
    """Check if this exact commit has already been reviewed (idempotency).

    Uses a Redis key `reviewed:{repo}:{pr}:{sha}` that is set after
    a review completes successfully.

    Args:
        redis: arq Redis connection pool.
        repo: Repository full name.
        pr: PR number.
        sha: Commit SHA.

    Returns:
        True if this commit was already reviewed.
    """
    key = f"reviewed:{repo}:{pr}:{sha}"
    return bool(await redis.exists(key))


async def mark_as_reviewed(redis: ArqRedis, repo: str, pr: int, sha: str) -> None:
    """Mark a commit as reviewed (idempotency marker).

    Set with a 24-hour TTL — old markers expire naturally.

    Args:
        redis: arq Redis connection pool.
        repo: Repository full name.
        pr: PR number.
        sha: Commit SHA.
    """
    key = f"reviewed:{repo}:{pr}:{sha}"
    await redis.set(key, "1", ex=86400)  # 24-hour TTL
