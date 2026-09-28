"""Unified diff parsing — converts raw patch text into structured FileContext objects.

Uses the `unidiff` library for correct hunk/line mapping that aligns with
the GitHub Review API's position-based inline comments.

References: PRD §4b (file filtering), §5 (diff parsing), §10 (language detection).
"""

from __future__ import annotations

import fnmatch

import structlog
from unidiff import PatchSet

from src.models.schemas import DiffContext, FileContext, HunkInfo, PRMetadata
from src.security.redaction import redact_secrets

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# §10 — Language detection by file extension
# ---------------------------------------------------------------------------

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".xml": "xml",
    ".proto": "protobuf",
    ".tf": "terraform",
    ".dockerfile": "dockerfile",
}

# ---------------------------------------------------------------------------
# §4b — File patterns to always skip (configurable via .reviewrules.yaml)
# ---------------------------------------------------------------------------

DEFAULT_SKIP_PATTERNS: list[str] = [
    "*.lock",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.generated.*",
    "*.pb.go",
    "*_pb2.py",
    "vendor/*",
    "node_modules/*",
    "__snapshots__/*",
    "*.svg",
    "*.png",
    "*.jpg",
    "*.gif",
    "*.ico",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "migrations/*.py",
]


def detect_language(filepath: str) -> str | None:
    """Detect programming language from file extension.

    Args:
        filepath: Relative file path (e.g. "src/utils/helpers.py").

    Returns:
        Language identifier string, or None if unknown.
    """
    # Handle compound extensions like .test.ts, .spec.js
    parts = filepath.rsplit("/", 1)[-1]  # Get filename only
    for ext in sorted(LANGUAGE_MAP.keys(), key=len, reverse=True):
        if parts.endswith(ext):
            return LANGUAGE_MAP[ext]
    return None


def should_skip_file(filepath: str, extra_skip_patterns: list[str] | None = None) -> bool:
    """Check if a file should be skipped based on skip patterns.

    Args:
        filepath: Relative file path.
        extra_skip_patterns: Additional patterns from .reviewrules.yaml.

    Returns:
        True if the file should be skipped.
    """
    patterns = DEFAULT_SKIP_PATTERNS + (extra_skip_patterns or [])
    return any(fnmatch.fnmatch(filepath, pattern) for pattern in patterns)


def parse_diff(
    patch_text: str,
    extra_skip_patterns: list[str] | None = None,
) -> list[FileContext]:
    """Parse a unified diff into structured FileContext objects.

    Args:
        patch_text: Raw unified diff / patch text.
        extra_skip_patterns: Additional file patterns to skip.

    Returns:
        List of FileContext objects, one per file in the diff (after filtering).
    """
    if not patch_text or not patch_text.strip():
        return []

    try:
        patch_set = PatchSet(patch_text)
    except Exception:
        logger.exception("diff.parse_error")
        return []

    files: list[FileContext] = []

    for patched_file in patch_set:
        filepath = patched_file.path

        # Apply skip patterns
        if should_skip_file(filepath, extra_skip_patterns):
            logger.debug("diff.skipped_file", file=filepath)
            continue

        # Build hunk info
        hunks: list[HunkInfo] = []
        for hunk in patched_file:
            added = [
                (line.target_line_no, line.value.rstrip("\n"))
                for line in hunk
                if line.is_added and line.target_line_no is not None
            ]
            removed = [
                (line.source_line_no, line.value.rstrip("\n"))
                for line in hunk
                if line.is_removed and line.source_line_no is not None
            ]
            hunks.append(
                HunkInfo(
                    source_start=hunk.source_start,
                    source_length=hunk.source_length,
                    target_start=hunk.target_start,
                    target_length=hunk.target_length,
                    added_lines=added,
                    removed_lines=removed,
                )
            )

        file_ctx = FileContext(
            path=filepath,
            language=detect_language(filepath),
            is_new=patched_file.is_added_file,
            is_deleted=patched_file.is_removed_file,
            is_renamed=patched_file.is_rename,
            hunks=hunks,
            patch=str(patched_file),
            additions=patched_file.added,
            deletions=patched_file.removed,
        )
        files.append(file_ctx)

    logger.info(
        "diff.parsed",
        total_files=len(patch_set),
        included_files=len(files),
        skipped_files=len(patch_set) - len(files),
    )

    return files


def build_diff_context(
    patch_text: str,
    metadata: PRMetadata,
    extra_skip_patterns: list[str] | None = None,
) -> DiffContext:
    """Build a complete DiffContext from raw patch text + PR metadata.

    This is the main entry point — called by the worker before invoking
    the LangGraph pipeline. Applies secret redaction BEFORE any agent
    sees the diff content (§15a).

    Args:
        patch_text: Raw unified diff text.
        metadata: PR metadata from the webhook payload.
        extra_skip_patterns: Additional patterns from .reviewrules.yaml.

    Returns:
        Fully assembled DiffContext with redacted content.
    """
    # Step 1: Redact secrets BEFORE parsing (agents never see raw secrets)
    redacted_text, redaction_records = redact_secrets(patch_text)

    # Step 2: Parse the redacted diff
    files = parse_diff(redacted_text, extra_skip_patterns)

    # Step 3: Compute totals
    total_additions = sum(f.additions for f in files)
    total_deletions = sum(f.deletions for f in files)

    diff_context = DiffContext(
        files=files,
        total_additions=total_additions,
        total_deletions=total_deletions,
        base_sha=metadata.base_sha,
        head_sha=metadata.head_sha,
        raw_patch=redacted_text,
        redaction_records=redaction_records,
    )

    logger.info(
        "diff.context_built",
        files=len(files),
        additions=total_additions,
        deletions=total_deletions,
        redactions=len(redaction_records),
    )

    return diff_context
