"""Run identity & checkpoint resume — deduplicate and resume reviews (Phase 7).

Before starting a review, compute a deterministic identity hash from:
  - The set of selected file paths (sorted)
  - The head commit SHA
  - The base commit SHA

This identity uniquely fingerprints the *inputs* to a review run. If a
previous run with the same identity completed successfully, the review
can be skipped entirely. If it completed partially (e.g. timed out),
the pipeline can resume from the LangGraph checkpoint.

The identity is stored in:
  - The DB `reviews` table (run_identity column) for persistence
  - The LangGraph checkpoint (thread_id = run_identity) for resumption
  - Redis (short-lived cache for fast skip checks)

Design rationale:
  - SHA-256 is used for collision resistance (important at scale).
  - Only *selected* files are included — excluded files (binaries, locks)
    don't change the review identity. This means a re-push that only
    changes a .lock file won't trigger a re-review.
  - The hash includes commit SHAs, so a force-push with different content
    always gets a fresh identity, even if the same files are selected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Run status enum
# ---------------------------------------------------------------------------

class RunStatus(StrEnum):
    """Status of a previous run with the same identity."""
    NOT_FOUND = "not_found"       # No previous run exists → run fresh
    COMPLETED = "completed"       # Previous run completed → skip
    PARTIAL = "partial"           # Previous run started but didn't finish → resume
    FAILED = "failed"             # Previous run failed → retry


# ---------------------------------------------------------------------------
# RunIdentity — the output of compute_run_identity()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunIdentity:
    """Deterministic fingerprint of a review run's inputs.

    Attributes:
        identity_hash: SHA-256 hex digest (64 chars) of the identity.
        short_hash: First 12 characters of identity_hash (for logging).
        head_sha: HEAD commit SHA of the PR.
        base_sha: Base commit SHA of the PR.
        selected_files: Sorted list of selected file paths.
        file_count: Number of selected files.
    """
    identity_hash: str
    short_hash: str
    head_sha: str
    base_sha: str
    selected_files: list[str]
    file_count: int

    def as_thread_id(self) -> str:
        """Use the identity hash as a LangGraph thread_id.

        This enables checkpoint-based resume: if the pipeline is
        re-invoked with the same thread_id, LangGraph resumes from
        the last checkpoint instead of restarting.
        """
        return f"review-{self.short_hash}"


# ---------------------------------------------------------------------------
# RunCheckResult — the output of check_previous_run()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunCheckResult:
    """Result of checking for a previous run with the same identity.

    Attributes:
        identity: The computed RunIdentity.
        status: Status of the previous run (if any).
        review_id: Database review ID of the previous run (if any).
        summary: Human-readable summary of the previous run (if completed).
        action: Recommended action: "skip", "resume", "retry", or "run".
    """
    identity: RunIdentity
    status: RunStatus
    review_id: str | None = None
    summary: str | None = None
    action: str = "run"


# ---------------------------------------------------------------------------
# Core: compute_run_identity
# ---------------------------------------------------------------------------

def compute_run_identity(
    selected_files: list[str],
    head_sha: str,
    base_sha: str,
) -> RunIdentity:
    """Compute a deterministic SHA-256 identity hash for a review run.

    The hash uniquely identifies the *inputs* to a review. If the same
    files are selected for the same commit range, the identity is the
    same — enabling skip/resume logic.

    Args:
        selected_files: List of selected file paths (from FileDecision).
            Order doesn't matter — they're sorted internally.
        head_sha: HEAD commit SHA of the PR.
        base_sha: Base commit SHA for the diff.

    Returns:
        RunIdentity with the computed hash and metadata.
    """
    # Sort files for deterministic ordering
    sorted_files = sorted(selected_files)

    # Build the identity payload as a canonical JSON string.
    # Using JSON ensures consistent serialization across runs.
    payload = json.dumps({
        "files": sorted_files,
        "head_sha": head_sha,
        "base_sha": base_sha,
    }, sort_keys=True, separators=(",", ":"))

    # SHA-256 for collision resistance
    identity_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    identity = RunIdentity(
        identity_hash=identity_hash,
        short_hash=identity_hash[:12],
        head_sha=head_sha,
        base_sha=base_sha,
        selected_files=sorted_files,
        file_count=len(sorted_files),
    )

    logger.info(
        "run_identity.computed",
        identity=identity.short_hash,
        files=identity.file_count,
        head_sha=head_sha[:8] if head_sha else "",
        base_sha=base_sha[:8] if base_sha else "",
    )

    return identity


def compute_run_identity_from_decisions(
    file_decisions: list,  # list[FileDecision]
    head_sha: str,
    base_sha: str,
) -> RunIdentity:
    """Convenience: compute identity from FileDecision objects.

    Extracts selected file paths from decisions, then delegates
    to compute_run_identity().

    Args:
        file_decisions: List of FileDecision objects.
        head_sha: HEAD commit SHA.
        base_sha: Base commit SHA.

    Returns:
        RunIdentity with the computed hash.
    """
    selected_paths = [
        d.file.path for d in file_decisions if d.selected
    ]
    return compute_run_identity(selected_paths, head_sha, base_sha)


# ---------------------------------------------------------------------------
# Database check: check_previous_run
# ---------------------------------------------------------------------------

async def check_previous_run(
    identity: RunIdentity,
    repo: str,
    pr_number: int,
) -> RunCheckResult:
    """Check if a previous run with this identity exists in the database.

    Queries the reviews table for a matching run_identity hash. Based on
    the status of the previous run, returns an action recommendation:
      - "skip":   Previous run completed → no need to re-review.
      - "resume": Previous run started but didn't finish → resume.
      - "retry":  Previous run failed → try again.
      - "run":    No previous run → start fresh.

    This function is designed to work with or without a database. If the
    DB is unavailable, it falls back to "run" (never blocks a review).

    Args:
        identity: Computed RunIdentity.
        repo: Repository full name.
        pr_number: PR number.

    Returns:
        RunCheckResult with the recommended action.
    """
    try:
        from src.db.connection import get_db_pool

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, status, completed_at
                FROM reviews
                WHERE repo = $1 AND pr_number = $2 AND run_identity = $3
                ORDER BY created_at DESC
                LIMIT 1
                """,
                repo, pr_number, identity.identity_hash,
            )

        if row is None:
            logger.info(
                "run_identity.no_previous_run",
                identity=identity.short_hash,
                repo=repo,
                pr=pr_number,
            )
            return RunCheckResult(
                identity=identity,
                status=RunStatus.NOT_FOUND,
                action="run",
            )

        review_id = str(row["id"])
        db_status = row["status"]

        if db_status in ("completed", "degraded"):
            logger.info(
                "run_identity.previous_completed",
                identity=identity.short_hash,
                review_id=review_id,
                status=db_status,
            )
            return RunCheckResult(
                identity=identity,
                status=RunStatus.COMPLETED,
                review_id=review_id,
                summary=f"Previous review {review_id[:8]} completed ({db_status})",
                action="skip",
            )

        if db_status == "failed":
            logger.info(
                "run_identity.previous_failed",
                identity=identity.short_hash,
                review_id=review_id,
            )
            return RunCheckResult(
                identity=identity,
                status=RunStatus.FAILED,
                review_id=review_id,
                action="retry",
            )

        # Status is "pending" or "processing" — partial run
        logger.info(
            "run_identity.previous_partial",
            identity=identity.short_hash,
            review_id=review_id,
            status=db_status,
        )
        return RunCheckResult(
            identity=identity,
            status=RunStatus.PARTIAL,
            review_id=review_id,
            action="resume",
        )

    except ImportError:
        logger.debug("run_identity.no_db_module")
        return RunCheckResult(
            identity=identity,
            status=RunStatus.NOT_FOUND,
            action="run",
        )

    except Exception as e:
        # DB unavailable — don't block the review
        logger.warning(
            "run_identity.db_check_failed",
            identity=identity.short_hash,
            error=str(e),
        )
        return RunCheckResult(
            identity=identity,
            status=RunStatus.NOT_FOUND,
            action="run",
        )


# ---------------------------------------------------------------------------
# Redis-based fast skip check
# ---------------------------------------------------------------------------

async def check_identity_in_cache(
    identity: RunIdentity,
    redis: object | None = None,
) -> bool:
    """Fast check if this identity was recently reviewed (Redis cache).

    This is a speed optimization — avoids the DB round-trip for the
    common case where a review just completed. The Redis key has a
    24-hour TTL (same as the debounce marker).

    Args:
        identity: Computed RunIdentity.
        redis: Redis connection (ArqRedis or compatible). If None, returns False.

    Returns:
        True if this identity was recently reviewed.
    """
    if redis is None:
        return False

    key = f"run_identity:{identity.identity_hash}"
    try:
        return bool(await redis.exists(key))  # type: ignore[union-attr]
    except Exception:
        return False


async def mark_identity_completed(
    identity: RunIdentity,
    redis: object | None = None,
    ttl_seconds: int = 86400,
) -> None:
    """Mark this identity as completed in the Redis cache.

    Called after a successful review to enable fast skip checks.

    Args:
        identity: Completed RunIdentity.
        redis: Redis connection.
        ttl_seconds: TTL for the cache entry (default: 24 hours).
    """
    if redis is None:
        return

    key = f"run_identity:{identity.identity_hash}"
    try:
        await redis.set(key, "1", ex=ttl_seconds)  # type: ignore[union-attr]
        logger.info(
            "run_identity.cached",
            identity=identity.short_hash,
            ttl=ttl_seconds,
        )
    except Exception as e:
        logger.warning(
            "run_identity.cache_failed",
            identity=identity.short_hash,
            error=str(e),
        )


def format_run_identity(identity: RunIdentity) -> str:
    """Format a concise identity string for logging.

    Example: "🔑 Run ID: a1b2c3d4e5f6 (3 files, abc123→main789)"
    """
    head_short = identity.head_sha[:7] if identity.head_sha else "?"
    base_short = identity.base_sha[:7] if identity.base_sha else "?"
    return (
        f"🔑 Run ID: {identity.short_hash} "
        f"({identity.file_count} files, {head_short}→{base_short})"
    )


def format_skip_message(check_result: RunCheckResult) -> str:
    """Format a message explaining why a review was skipped/resumed.

    Args:
        check_result: Result from check_previous_run().

    Returns:
        Human-readable message, or empty string if action is "run".
    """
    if check_result.action == "skip":
        return (
            f"⏭️ Skipping review — identical inputs already reviewed "
            f"({check_result.identity.short_hash}). "
            f"{check_result.summary or ''}"
        )
    elif check_result.action == "resume":
        return (
            f"🔄 Resuming partial review ({check_result.identity.short_hash}). "
            f"Previous run was interrupted."
        )
    elif check_result.action == "retry":
        return (
            f"🔄 Retrying review ({check_result.identity.short_hash}). "
            f"Previous run failed."
        )
    return ""
