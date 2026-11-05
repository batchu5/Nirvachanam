"""Deterministic comment line resolution — Phase 4.

After the agent loop produces RawComment objects with verbatim code snippets
(``existing_code``), this module resolves them to exact line numbers in the
diff.  No LLM is used — the entire process is deterministic string matching.

Resolution strategies (tried in order):
  1. **Exact match** — ``existing_code`` appears verbatim in a diff hunk's
     added/context lines.
  2. **Normalized match** — strip whitespace and compare.
  3. **Multi-line window** — match each line of a multi-line snippet
     against consecutive hunk lines.
  4. **Patch scan** — walk the raw patch text and track line numbers.
  5. **Cross-file relocation** — if the snippet doesn't match the
     filed-against file, search ALL diffs for a unique match and re-file
     the comment.
  6. **Fallback** — first added line in the target file's diff.

Design notes:
  - The resolver NEVER modifies ``comment.message`` or other fields.
    It only sets ``line`` (and optionally ``end_line``) and may update
    ``file`` during cross-file relocation.
  - All matching is case-sensitive (code is case-sensitive).
  - When a snippet matches multiple locations, the resolver picks the
    first match (closest to the top of the diff).

References: Alibaba OCR resolver.go
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from src.agents.tools.definitions import RawComment
from src.models.schemas import DiffContext, FileContext, HunkInfo

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------

@dataclass
class ResolutionResult:
    """Result of resolving a single comment's line number.

    Attributes:
        line: Resolved 1-indexed line number in the target file.
        end_line: Optional end line for multi-line snippets.
        file: Resolved file path (may differ from original if relocated).
        strategy: Which resolution strategy succeeded.
        relocated: True if the comment was moved to a different file.
        original_file: Original file path before relocation.
    """
    line: int
    end_line: int | None = None
    file: str = ""
    strategy: str = "fallback"
    relocated: bool = False
    original_file: str = ""


# ---------------------------------------------------------------------------
# Batch resolution — main entry point
# ---------------------------------------------------------------------------

def resolve_line_numbers(
    comments: list[RawComment],
    diff_context: DiffContext,
) -> list[tuple[RawComment, ResolutionResult]]:
    """Resolve line numbers for all comments against the diff.

    Primary: match comment ``existing_code`` snippets against diff hunks.
    Fallback: scan full patch text line-by-line.

    Args:
        comments: RawComment objects from the agent loop.
        diff_context: Full parsed diff context.

    Returns:
        List of (comment, resolution) pairs.  The resolution contains the
        resolved line number and metadata about which strategy succeeded.
    """
    file_lookup = {f.path: f for f in diff_context.files}
    results: list[tuple[RawComment, ResolutionResult]] = []

    for comment in comments:
        resolution = _resolve_single_comment(
            comment, diff_context, file_lookup,
        )
        results.append((comment, resolution))

    _log_resolution_stats(results)
    return results


def _resolve_single_comment(
    comment: RawComment,
    diff_context: DiffContext,
    file_lookup: dict[str, FileContext],
) -> ResolutionResult:
    """Resolve a single comment to a line number.

    Tries strategies in order of precision.
    """
    snippet = (comment.existing_code or "").strip()
    file_path = comment.file

    # Strategy 0: no snippet → immediate fallback
    if not snippet:
        line = _fallback_line(file_path, file_lookup)
        return ResolutionResult(
            line=line, file=file_path, strategy="no_snippet",
        )

    # Try resolution in the filed-against file first
    file_ctx = file_lookup.get(file_path)
    if file_ctx is not None:
        result = _resolve_in_file(snippet, file_ctx)
        if result is not None:
            result.file = file_path
            return result

    # Strategy 5: cross-file relocation
    relocated = relocate_across_files(comment, diff_context.files)
    if relocated is not None:
        return relocated

    # Last resort fallback
    line = _fallback_line(file_path, file_lookup)
    return ResolutionResult(
        line=line, file=file_path, strategy="fallback",
    )


# ---------------------------------------------------------------------------
# In-file resolution strategies
# ---------------------------------------------------------------------------

def _resolve_in_file(
    snippet: str,
    file_ctx: FileContext,
) -> ResolutionResult | None:
    """Try all in-file resolution strategies for a snippet."""

    # Strategy 1: exact match in hunk added lines
    result = _match_exact_hunk(snippet, file_ctx)
    if result is not None:
        return result

    # Strategy 2: normalized (whitespace-insensitive) match
    result = _match_normalized_hunk(snippet, file_ctx)
    if result is not None:
        return result

    # Strategy 3: multi-line window match
    snippet_lines = snippet.splitlines()
    if len(snippet_lines) > 1:
        result = _match_multiline_window(snippet_lines, file_ctx)
        if result is not None:
            return result

    # Strategy 4: patch text scan
    result = _match_patch_scan(snippet, file_ctx)
    if result is not None:
        return result

    return None


def _match_exact_hunk(
    snippet: str,
    file_ctx: FileContext,
) -> ResolutionResult | None:
    """Strategy 1: exact substring match in hunk added/context lines."""
    for hunk in file_ctx.hunks:
        # Check added lines
        for line_no, line_content in hunk.added_lines:
            stripped = line_content.strip()
            if not stripped:
                continue
            if snippet in stripped or stripped in snippet:
                return ResolutionResult(
                    line=line_no, strategy="exact_hunk_added",
                )

    return None


def _match_normalized_hunk(
    snippet: str,
    file_ctx: FileContext,
) -> ResolutionResult | None:
    """Strategy 2: whitespace-normalized match.

    Collapses all runs of whitespace to a single space before comparing.
    Catches cases where the LLM slightly reformatted the code.
    """
    normalized_snippet = _normalize_ws(snippet)
    if not normalized_snippet:
        return None

    for hunk in file_ctx.hunks:
        for line_no, line_content in hunk.added_lines:
            if _normalize_ws(line_content) == normalized_snippet:
                return ResolutionResult(
                    line=line_no, strategy="normalized_hunk",
                )
            # Also try substring containment
            if normalized_snippet in _normalize_ws(line_content):
                return ResolutionResult(
                    line=line_no, strategy="normalized_hunk_substring",
                )

    return None


def _match_multiline_window(
    snippet_lines: list[str],
    file_ctx: FileContext,
) -> ResolutionResult | None:
    """Strategy 3: match consecutive lines of a multi-line snippet.

    Builds a window of added lines from each hunk and checks if the
    snippet lines appear as a contiguous subsequence.
    """
    # Normalize snippet lines
    normalized_snippet = [_normalize_ws(l) for l in snippet_lines if l.strip()]
    if not normalized_snippet:
        return None

    for hunk in file_ctx.hunks:
        # Build list of (line_no, normalized_content) for added lines
        added = [
            (ln, _normalize_ws(content))
            for ln, content in hunk.added_lines
            if content.strip()
        ]

        # Sliding window match
        window_size = len(normalized_snippet)
        for i in range(len(added) - window_size + 1):
            window = [content for _, content in added[i:i + window_size]]
            if window == normalized_snippet:
                start_line = added[i][0]
                end_line = added[i + window_size - 1][0]
                return ResolutionResult(
                    line=start_line,
                    end_line=end_line,
                    strategy="multiline_window",
                )

        # Partial match: just the first line of the snippet
        first_norm = normalized_snippet[0]
        for ln, content in added:
            if first_norm == content or first_norm in content:
                return ResolutionResult(
                    line=ln, strategy="multiline_first_line",
                )

    return None


def _match_patch_scan(
    snippet: str,
    file_ctx: FileContext,
) -> ResolutionResult | None:
    """Strategy 4: walk the raw patch text tracking line numbers.

    Parses @@ hunk headers for line numbers and walks each line,
    looking for the snippet in added/context lines.
    """
    if not file_ctx.patch:
        return None

    normalized_snippet = _normalize_ws(snippet)
    current_line = 1

    for patch_line in file_ctx.patch.splitlines():
        # Parse hunk header
        if patch_line.startswith("@@"):
            m = re.search(r'\+(\d+)', patch_line)
            if m:
                current_line = int(m.group(1))
            continue

        # Skip removed lines (they don't exist in the target file)
        if patch_line.startswith("-"):
            continue

        # Match against added or context lines
        clean = patch_line.lstrip("+").strip() if patch_line.startswith("+") else patch_line.strip()
        if clean and (snippet in clean or normalized_snippet == _normalize_ws(clean)):
            return ResolutionResult(
                line=current_line, strategy="patch_scan",
            )

        # Advance line counter for non-removed lines
        if not patch_line.startswith("-"):
            current_line += 1

    return None


# ---------------------------------------------------------------------------
# Cross-file relocation
# ---------------------------------------------------------------------------

def relocate_across_files(
    comment: RawComment,
    all_files: list[FileContext],
) -> ResolutionResult | None:
    """If ``existing_code`` doesn't match the filed-against file,
    search ALL diffs for a unique match and re-file the comment.

    Only relocates if there is exactly ONE match across all files
    (to avoid ambiguity).

    Args:
        comment: The comment to relocate.
        all_files: All FileContext objects in the diff.

    Returns:
        ResolutionResult with ``relocated=True`` if a unique match was
        found in a different file, or None.
    """
    snippet = (comment.existing_code or "").strip()
    if not snippet:
        return None

    original_file = comment.file
    matches: list[tuple[str, int]] = []  # (file_path, line_no)

    for file_ctx in all_files:
        if file_ctx.path == original_file:
            continue  # Already tried this file

        # Try exact match in hunk added lines
        for hunk in file_ctx.hunks:
            for line_no, line_content in hunk.added_lines:
                if snippet in line_content.strip() or line_content.strip() in snippet:
                    matches.append((file_ctx.path, line_no))
                    break  # One match per file is enough
            if matches and matches[-1][0] == file_ctx.path:
                break

        # Also try normalized match
        if not any(m[0] == file_ctx.path for m in matches):
            norm_snippet = _normalize_ws(snippet)
            for hunk in file_ctx.hunks:
                for line_no, line_content in hunk.added_lines:
                    if norm_snippet == _normalize_ws(line_content):
                        matches.append((file_ctx.path, line_no))
                        break
                if matches and matches[-1][0] == file_ctx.path:
                    break

    if len(matches) == 1:
        # Unique match in a different file — relocate
        target_file, target_line = matches[0]
        logger.info(
            "resolver.relocated_comment",
            original_file=original_file,
            target_file=target_file,
            target_line=target_line,
            snippet_preview=snippet[:80],
        )
        return ResolutionResult(
            line=target_line,
            file=target_file,
            strategy="cross_file_relocation",
            relocated=True,
            original_file=original_file,
        )

    if len(matches) > 1:
        logger.debug(
            "resolver.ambiguous_relocation",
            original_file=original_file,
            candidate_count=len(matches),
            candidates=[m[0] for m in matches[:5]],
        )

    return None


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_line(
    file_path: str,
    file_lookup: dict[str, FileContext],
) -> int:
    """Return the first added line in the file, or 1 as last resort."""
    file_ctx = file_lookup.get(file_path)
    if file_ctx is not None:
        for hunk in file_ctx.hunks:
            if hunk.added_lines:
                return hunk.added_lines[0][0]
            return hunk.target_start
    return 1


# ---------------------------------------------------------------------------
# Batch application — mutate comments in place
# ---------------------------------------------------------------------------

def apply_resolutions(
    comments: list[RawComment],
    diff_context: DiffContext,
) -> list[ResolvedComment]:
    """Resolve line numbers and return enriched ResolvedComment objects.

    This is the convenience entry point that combines resolution + output.

    Args:
        comments: RawComment objects from the agent loop.
        diff_context: Full parsed diff context.

    Returns:
        List of ResolvedComment with line numbers and metadata.
    """
    pairs = resolve_line_numbers(comments, diff_context)
    return [
        ResolvedComment(
            comment=comment,
            line=resolution.line,
            end_line=resolution.end_line,
            file=resolution.file or comment.file,
            strategy=resolution.strategy,
            relocated=resolution.relocated,
            original_file=resolution.original_file,
        )
        for comment, resolution in pairs
    ]


# ---------------------------------------------------------------------------
# ResolvedComment — comment with resolved line number
# ---------------------------------------------------------------------------

@dataclass
class ResolvedComment:
    """A RawComment with a resolved line number.

    This is the output of the resolver. It contains the original comment
    plus the resolved file/line info.

    Attributes:
        comment: The original RawComment.
        line: Resolved 1-indexed line number.
        end_line: Optional end line for multi-line findings.
        file: Resolved file path (may differ from comment.file).
        strategy: Which resolution strategy succeeded.
        relocated: True if the comment was moved to a different file.
        original_file: Original file path before relocation.
    """
    comment: RawComment
    line: int
    end_line: int | None = None
    file: str = ""
    strategy: str = "fallback"
    relocated: bool = False
    original_file: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_ws(s: str) -> str:
    """Collapse all whitespace runs to a single space and strip."""
    return re.sub(r'\s+', ' ', s).strip()


def _log_resolution_stats(
    results: list[tuple[RawComment, ResolutionResult]],
) -> None:
    """Log summary statistics about resolution outcomes."""
    if not results:
        return

    strategy_counts: dict[str, int] = {}
    relocated_count = 0
    for _, resolution in results:
        strategy_counts[resolution.strategy] = (
            strategy_counts.get(resolution.strategy, 0) + 1
        )
        if resolution.relocated:
            relocated_count += 1

    logger.info(
        "resolver.stats",
        total=len(results),
        strategies=strategy_counts,
        relocated=relocated_count,
    )


# ---------------------------------------------------------------------------
# Convenience: format resolution summary
# ---------------------------------------------------------------------------

def format_resolution_summary(
    results: list[tuple[RawComment, ResolutionResult]],
) -> str:
    """Format a human-readable summary of resolution outcomes.

    Example: "🎯 Resolved 5 comments: 3 exact, 1 normalized, 1 fallback"
    """
    if not results:
        return "🎯 No comments to resolve"

    strategy_counts: dict[str, int] = {}
    for _, resolution in results:
        label = resolution.strategy.replace("_", " ")
        strategy_counts[label] = strategy_counts.get(label, 0) + 1

    parts = [f"{count} {strategy}" for strategy, count in strategy_counts.items()]
    return f"🎯 Resolved {len(results)} comments: {', '.join(parts)}"
