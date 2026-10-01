"""Deferred backfill jobs for quota-skipped agents.

When an agent is skipped due to quota exhaustion, a backfill job is
enqueued to retry just that agent later and patch the existing review.

References: PRD §1c (deferred backfill).
"""

from __future__ import annotations

from datetime import timedelta

import structlog
from arq.connections import ArqRedis

logger = structlog.get_logger()


async def enqueue_backfill(
    redis: ArqRedis,
    repo: str,
    pr: int,
    sha: str,
    skipped_agent: str,
    review_id: str,
    retry_after: int = 120,
) -> None:
    """Enqueue a deferred job to backfill a quota-skipped agent.

    The job will fire after `retry_after + 30` seconds (after quota resets),
    re-run just the skipped agent, run critic on the new findings, and
    PATCH the existing posted review.

    Args:
        redis: ArqRedis connection.
        repo: Repository full name (owner/repo).
        pr: PR number.
        sha: Commit SHA the review was for.
        skipped_agent: Name of the agent that was skipped.
        review_id: Database ID of the existing review to patch.
        retry_after: Seconds until the provider quota resets.
    """
    job_id = f"backfill:{repo}:{pr}:{sha}:{skipped_agent}"

    await redis.enqueue_job(
        "backfill_agent_review",
        repo=repo,
        pr=pr,
        sha=sha,
        agent=skipped_agent,
        review_id=review_id,
        _defer_by=timedelta(seconds=retry_after + 30),
        _job_id=job_id,  # Idempotent — won't double-enqueue
    )

    logger.info(
        "backfill.enqueued",
        job_id=job_id,
        agent=skipped_agent,
        defer_seconds=retry_after + 30,
    )
