"""Database operations — CRUD for reviews and findings.

Provides typed async functions for interacting with the reviews and findings
tables. Uses the shared asyncpg connection pool from src.db.connection.

References: PRD §13 (database schema).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog

from src.db.connection import get_db_pool
from src.models.schemas import Finding, TokenUsage

logger = structlog.get_logger()


async def create_review(
    repo: str,
    pr_number: int,
    commit_sha: str,
    base_sha: str | None = None,
    status: str = "pending",
) -> str:
    """Create a new review record.

    Args:
        repo: Repository full name.
        pr_number: PR number.
        commit_sha: HEAD commit SHA being reviewed.
        base_sha: Base SHA for the diff.
        status: Initial status.

    Returns:
        UUID string of the created review.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO reviews (repo, pr_number, commit_sha, base_sha, status)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (repo, pr_number, commit_sha) DO UPDATE
            SET status = $5
            RETURNING id
            """,
            repo, pr_number, commit_sha, base_sha, status,
        )
        review_id = str(row["id"])
        logger.info(
            "db.review_created",
            review_id=review_id,
            repo=repo,
            pr=pr_number,
            sha=commit_sha[:8],
        )
        return review_id


async def update_review_status(
    review_id: str,
    status: str,
    degraded_agents: list[str] | None = None,
    model_config: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> None:
    """Update a review's status and metadata.

    Args:
        review_id: UUID of the review.
        status: New status (completed, failed, degraded).
        degraded_agents: List of agents that were skipped.
        model_config: Which models were used per agent.
        token_usage: Per-agent token consumption.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE reviews
            SET status = $2,
                degraded_agents = $3,
                model_config = $4,
                token_usage = $5,
                completed_at = CASE WHEN $2 IN ('completed', 'failed', 'degraded')
                    THEN now() ELSE completed_at END
            WHERE id = $1::uuid
            """,
            review_id,
            status,
            degraded_agents,
            json.dumps(model_config) if model_config else None,
            json.dumps(token_usage) if token_usage else None,
        )
        logger.info("db.review_updated", review_id=review_id, status=status)


async def store_findings(
    review_id: str,
    findings: list[Finding],
) -> list[str]:
    """Store findings in the database.

    Args:
        review_id: UUID of the parent review.
        findings: List of Finding objects to store.

    Returns:
        List of UUID strings for the stored findings.
    """
    if not findings:
        return []

    pool = await get_db_pool()
    finding_ids: list[str] = []

    async with pool.acquire() as conn:
        for finding in findings:
            row = await conn.fetchrow(
                """
                INSERT INTO findings (
                    review_id, schema_version, file, line, end_line,
                    severity, category, message, suggested_fix,
                    confidence, agent, language, content_signature
                )
                VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id
                """,
                review_id,
                finding.schema_version,
                finding.file,
                finding.line,
                finding.end_line,
                finding.severity,
                finding.category,
                finding.message,
                finding.suggested_fix,
                finding.confidence,
                finding.agent,
                finding.language,
                finding.content_signature,
            )
            finding_ids.append(str(row["id"]))

    logger.info(
        "db.findings_stored",
        review_id=review_id,
        count=len(finding_ids),
    )
    return finding_ids


async def get_latest_review(
    repo: str,
    pr_number: int,
) -> dict[str, Any] | None:
    """Get the latest review for a PR.

    Args:
        repo: Repository full name.
        pr_number: PR number.

    Returns:
        Review record as dict, or None if no reviews exist.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, repo, pr_number, commit_sha, base_sha, status,
                   degraded_agents, created_at, completed_at
            FROM reviews
            WHERE repo = $1 AND pr_number = $2
            ORDER BY created_at DESC
            LIMIT 1
            """,
            repo, pr_number,
        )
        if row:
            return dict(row)
        return None


async def check_review_exists(
    repo: str,
    pr_number: int,
    commit_sha: str,
) -> bool:
    """Check if a review already exists for this exact commit.

    Args:
        repo: Repository full name.
        pr_number: PR number.
        commit_sha: Commit SHA.

    Returns:
        True if a review exists for this commit.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 1 FROM reviews
            WHERE repo = $1 AND pr_number = $2 AND commit_sha = $3
            AND status IN ('completed', 'degraded')
            """,
            repo, pr_number, commit_sha,
        )
        return row is not None
