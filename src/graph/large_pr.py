"""Large PR handling — file filtering, chunking, and risk-based prioritization.

Implements PRD §4: size tiers, automatic file filtering, and chunking
for PRs that exceed single-call context limits.

References: PRD §4a (size tiers), §4b (file filtering), §4c (chunking).
"""

from __future__ import annotations

import fnmatch
from enum import StrEnum

import structlog

from src.models.schemas import DiffContext, FileContext

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# §4a — PR size tiers
# ---------------------------------------------------------------------------

class PRSizeTier(StrEnum):
    """PR size classification — drives chunking and filtering strategy."""
    SMALL = "small"       # 1-10 files: full review
    MEDIUM = "medium"     # 11-50 files: filter noise, prioritize
    LARGE = "large"       # 51-200 files: chunk into batches
    MEGA = "mega"         # 200+ files: top 30 only


def classify_pr_size(file_count: int) -> PRSizeTier:
    """Classify a PR by number of changed files."""
    if file_count <= 10:
        return PRSizeTier.SMALL
    elif file_count <= 50:
        return PRSizeTier.MEDIUM
    elif file_count <= 200:
        return PRSizeTier.LARGE
    else:
        return PRSizeTier.MEGA


# ---------------------------------------------------------------------------
# §4b — File filtering (security-sensitive files first)
# ---------------------------------------------------------------------------

# Risk score by file extension — higher = review first
RISK_SCORES: dict[str, int] = {
    ".py": 8,
    ".js": 8,
    ".ts": 8,
    ".jsx": 7,
    ".tsx": 7,
    ".sql": 9,        # SQL injection risk
    ".yaml": 6,       # Config / secrets
    ".yml": 6,
    ".env": 10,       # Secrets
    ".toml": 5,
    ".go": 8,
    ".java": 8,
    ".rs": 7,
    ".rb": 7,
    ".php": 8,
    ".sh": 6,         # Shell scripts
    ".dockerfile": 5,
    ".tf": 6,         # Terraform
}


def _get_risk_score(path: str) -> int:
    """Get risk score for a file based on its extension."""
    lower = path.lower()
    for ext, score in sorted(RISK_SCORES.items(), key=lambda x: -len(x[0])):
        if lower.endswith(ext):
            return score
    return 3  # Default low-risk score


def sort_files_by_risk(files: list[FileContext]) -> list[FileContext]:
    """Sort files by risk score (highest risk first), then by additions."""
    return sorted(
        files,
        key=lambda f: (_get_risk_score(f.path), f.additions),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# §4c — Chunking strategy
# ---------------------------------------------------------------------------

MAX_FILES_PER_CHUNK = 15
MAX_LINES_PER_FILE = 200  # Truncate individual files beyond this
MAX_FILES_MEGA = 30        # Only review top N files for mega PRs


def chunk_files(
    files: list[FileContext],
    max_per_chunk: int = MAX_FILES_PER_CHUNK,
) -> list[list[FileContext]]:
    """Split file list into review-sized chunks.

    Groups files by directory when possible to keep related files together.

    Args:
        files: Sorted list of FileContext objects.
        max_per_chunk: Maximum files per chunk.

    Returns:
        List of file chunks.
    """
    if not files:
        return []

    # Group by top-level directory
    dir_groups: dict[str, list[FileContext]] = {}
    for f in files:
        parts = f.path.split("/")
        top_dir = parts[0] if len(parts) > 1 else ""
        dir_groups.setdefault(top_dir, []).append(f)

    chunks: list[list[FileContext]] = []
    current_chunk: list[FileContext] = []

    for _dir, group_files in sorted(dir_groups.items()):
        for f in group_files:
            if len(current_chunk) >= max_per_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            current_chunk.append(f)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def truncate_file_patch(file_ctx: FileContext, max_lines: int = MAX_LINES_PER_FILE) -> FileContext:
    """Truncate a file's patch to max_lines if it's too long.

    Adds a warning comment at the truncation point.
    """
    lines = file_ctx.patch.splitlines()
    if len(lines) <= max_lines:
        return file_ctx

    truncated_patch = "\n".join(lines[:max_lines])
    truncated_patch += f"\n... [TRUNCATED: {len(lines) - max_lines} more lines]"

    logger.info(
        "large_pr.file_truncated",
        file=file_ctx.path,
        original_lines=len(lines),
        truncated_to=max_lines,
    )

    return file_ctx.model_copy(update={"patch": truncated_patch})


def prepare_diff_for_review(diff_context: DiffContext) -> tuple[DiffContext, PRSizeTier, str]:
    """Apply large-PR handling strategy to a DiffContext.

    Args:
        diff_context: Original parsed diff.

    Returns:
        Tuple of (processed_diff_context, size_tier, user_message).
        user_message contains a warning if the PR was truncated/filtered.
    """
    tier = classify_pr_size(len(diff_context.files))
    message = ""

    if tier == PRSizeTier.SMALL:
        return diff_context, tier, message

    # Sort by risk for all non-small PRs
    sorted_files = sort_files_by_risk(diff_context.files)

    if tier == PRSizeTier.MEGA:
        # Only review top N highest-risk files
        sorted_files = sorted_files[:MAX_FILES_MEGA]
        message = (
            f"📦 This PR has {len(diff_context.files)} files. "
            f"Reviewing the top {MAX_FILES_MEGA} highest-risk files. "
            f"Add a `.reviewrules.yaml` to customize."
        )
        logger.info(
            "large_pr.mega_truncated",
            total_files=len(diff_context.files),
            reviewing=len(sorted_files),
        )

    # Truncate individual large files
    processed_files = [truncate_file_patch(f) for f in sorted_files]

    # Build new diff context
    processed = diff_context.model_copy(update={
        "files": processed_files,
        "total_additions": sum(f.additions for f in processed_files),
        "total_deletions": sum(f.deletions for f in processed_files),
    })

    if tier == PRSizeTier.MEDIUM:
        message = f"Reviewing {len(processed_files)} files (filtered and prioritized by risk)."
    elif tier == PRSizeTier.LARGE:
        message = (
            f"📦 Large PR ({len(diff_context.files)} files). "
            f"Processing in batches of {MAX_FILES_PER_CHUNK}."
        )

    logger.info(
        "large_pr.prepared",
        tier=tier,
        original_files=len(diff_context.files),
        processed_files=len(processed_files),
    )

    return processed, tier, message
