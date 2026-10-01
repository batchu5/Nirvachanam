"""Rule-based PR triage — classifies PRs into review tiers with ZERO LLM tokens.

Implements Strategy 2: tiered review system. The triage step is pure rules —
no LLM calls. It checks file count, total line changes, and file types to
determine the appropriate review depth.

Now works with the file selection layer (Phase 1): the selection step decides
WHAT to review, triage decides HOW DEEP. Triage only sees pre-filtered
FileDecision objects, not raw DiffContext.files.

Review tiers:
  - FAST:     Small, low-risk PRs (< 50 lines, no security-sensitive files)
  - STANDARD: Most PRs — single comprehensive LLM call + rule-based critic
  - DEEP:     Large or security-critical PRs — comprehensive review with critic

References: Token optimization strategy §2 (tiered review system).
"""

from __future__ import annotations

from enum import StrEnum

import structlog

from src.models.schemas import DiffContext, FileContext

logger = structlog.get_logger()


class ReviewTier(StrEnum):
    """Review depth tiers — drives prompt selection and pipeline complexity."""
    FAST = "fast"           # ~2,000 tokens — critical issues only
    STANDARD = "standard"   # ~5,000 tokens — full review + rule-based critic
    DEEP = "deep"           # ~8,000 tokens — comprehensive + rule-based critic


# ---------------------------------------------------------------------------
# Security-sensitive file patterns
# ---------------------------------------------------------------------------

SECURITY_SENSITIVE_EXTENSIONS = frozenset({
    ".env", ".pem", ".key", ".crt", ".p12", ".pfx",
    ".sql", ".yaml", ".yml", ".toml",
})

SECURITY_SENSITIVE_PATTERNS = (
    "auth", "login", "password", "secret", "credential",
    "token", "session", "oauth", "jwt", "crypto", "encrypt",
    "payment", "billing", "stripe", "paypal",
    "admin", "permission", "rbac", "acl",
    "middleware", "guard", "policy",
    "docker", "kubernetes", "k8s", "terraform", ".tf",
    "migration", "schema",
)


# ---------------------------------------------------------------------------
# Security detection helpers — used by both triage_pr and triage_from_decisions
# ---------------------------------------------------------------------------

def _is_security_sensitive_path(path: str) -> bool:
    """Check if a single file path is security-sensitive."""
    path_lower = path.lower()

    # Check extension
    for ext in SECURITY_SENSITIVE_EXTENSIONS:
        if path_lower.endswith(ext):
            return True

    # Check path patterns
    for pattern in SECURITY_SENSITIVE_PATTERNS:
        if pattern in path_lower:
            return True

    return False


def _has_security_sensitive_files_in_list(files: list[FileContext]) -> bool:
    """Check if any file in a list is security-sensitive."""
    return any(_is_security_sensitive_path(f.path) for f in files)


def _is_docs_only_files(files: list[FileContext]) -> bool:
    """Check if the file list only contains documentation files."""
    doc_extensions = {".md", ".rst", ".txt", ".adoc", ".rdoc"}
    for file_ctx in files:
        ext = file_ctx.path.lower()
        dot_pos = ext.rfind(".")
        if dot_pos != -1:
            ext = ext[dot_pos:]
        else:
            ext = ""
        if ext not in doc_extensions:
            return False
    return True


# ---------------------------------------------------------------------------
# NEW: Triage from FileDecisions (Phase 1 integration)
# ---------------------------------------------------------------------------

def triage_from_decisions(
    file_decisions: list,  # list[FileDecision]
) -> tuple[ReviewTier, str]:
    """Classify a PR into a review tier using pre-filtered file decisions.

    This is the NEW primary entry point that works with the file selection
    layer. It only considers files that were SELECTED for review (not
    excluded ones), giving cleaner tier classification.

    Uses ZERO LLM tokens — pure rule-based logic.

    Args:
        file_decisions: List of FileDecision objects from select_files().

    Returns:
        Tuple of (ReviewTier, reason_string).
    """
    # Import here to avoid circular dependency
    from src.agents.file_selection import FileDecision

    # Extract only selected files (ExcludeReason.NONE)
    selected = [d for d in file_decisions if d.selected]

    if not selected:
        return ReviewTier.FAST, "No files selected for review"

    selected_file_contexts = [d.file for d in selected]

    # Docs-only check on selected files
    if _is_docs_only_files(selected_file_contexts):
        return ReviewTier.FAST, "Documentation-only changes"

    # Count reviewable metrics from selected files only
    reviewable_lines = sum(
        d.file.additions + d.file.deletions for d in selected
    )
    reviewable_files = len(selected)
    total_tokens = sum(d.diff_tokens for d in selected)
    has_security = _has_security_sensitive_files_in_list(selected_file_contexts)

    logger.info(
        "triage.analysis",
        reviewable_lines=reviewable_lines,
        reviewable_files=reviewable_files,
        total_estimated_tokens=total_tokens,
        has_security_files=has_security,
    )

    # DEEP tier: security-critical or large PRs
    if has_security and reviewable_lines > 100:
        return ReviewTier.DEEP, (
            f"Security-sensitive files with {reviewable_lines} line changes"
        )

    if reviewable_files > 15 or reviewable_lines > 500:
        return ReviewTier.DEEP, (
            f"Large PR: {reviewable_files} files, {reviewable_lines} lines"
        )

    # FAST tier: small, low-risk PRs
    if reviewable_lines < 50 and not has_security:
        return ReviewTier.FAST, (
            f"Small PR: {reviewable_lines} lines, no security-sensitive files"
        )

    # Everything else: STANDARD
    return ReviewTier.STANDARD, (
        f"Standard PR: {reviewable_files} files, {reviewable_lines} lines"
    )


# ---------------------------------------------------------------------------
# Legacy: Triage from DiffContext (backward compatibility)
# ---------------------------------------------------------------------------

# File patterns to skip entirely (no review needed) — legacy, use
# file_selection.py DEFAULT_SKIP_PATTERNS for new code
SKIP_PATTERNS = (
    ".lock", ".min.js", ".min.css", ".map",
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", "__pycache__",
)


def _count_reviewable_lines(diff_context: DiffContext) -> int:
    """Count total added + deleted lines, excluding skip-pattern files."""
    total = 0
    for file_ctx in diff_context.files:
        path_lower = file_ctx.path.lower()
        if any(path_lower.endswith(skip) for skip in SKIP_PATTERNS):
            continue
        total += file_ctx.additions + file_ctx.deletions
    return total


def _count_reviewable_files(diff_context: DiffContext) -> int:
    """Count files that are actually worth reviewing."""
    count = 0
    for file_ctx in diff_context.files:
        path_lower = file_ctx.path.lower()
        if any(path_lower.endswith(skip) for skip in SKIP_PATTERNS):
            continue
        if file_ctx.is_deleted:
            continue
        count += 1
    return count


def triage_pr(diff_context: DiffContext) -> tuple[ReviewTier, str]:
    """Classify a PR into a review tier using pure rule-based logic.

    LEGACY entry point — use triage_from_decisions() when file_decisions
    are available. This function is kept for backward compatibility and
    for contexts where the file selection layer hasn't run yet.

    This function uses ZERO LLM tokens. It examines:
    - Total reviewable line changes
    - Number of reviewable files
    - Presence of security-sensitive files
    - Whether it's a docs-only PR

    Args:
        diff_context: Parsed diff context from the PR.

    Returns:
        Tuple of (ReviewTier, reason_string).
    """
    # Edge case: empty diff
    if not diff_context.files:
        return ReviewTier.FAST, "Empty diff — nothing to review"

    # Docs-only PRs get fast path
    if _is_docs_only_files(diff_context.files):
        return ReviewTier.FAST, "Documentation-only changes"

    reviewable_lines = _count_reviewable_lines(diff_context)
    reviewable_files = _count_reviewable_files(diff_context)
    has_security = _has_security_sensitive_files_in_list(diff_context.files)

    logger.info(
        "triage.analysis",
        reviewable_lines=reviewable_lines,
        reviewable_files=reviewable_files,
        has_security_files=has_security,
    )

    # DEEP tier: security-critical or large PRs
    if has_security and reviewable_lines > 100:
        return ReviewTier.DEEP, (
            f"Security-sensitive files with {reviewable_lines} line changes"
        )

    if reviewable_files > 15 or reviewable_lines > 500:
        return ReviewTier.DEEP, (
            f"Large PR: {reviewable_files} files, {reviewable_lines} lines"
        )

    # FAST tier: small, low-risk PRs
    if reviewable_lines < 50 and not has_security:
        return ReviewTier.FAST, (
            f"Small PR: {reviewable_lines} lines, no security-sensitive files"
        )

    # Everything else: STANDARD
    return ReviewTier.STANDARD, (
        f"Standard PR: {reviewable_files} files, {reviewable_lines} lines"
    )
