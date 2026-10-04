"""Semantic file grouping — groups related files for review (Phase 5).

For small PRs, all selected files go in a single group (no LLM needed).
For larger PRs, files are grouped by semantic relationship using an LLM
call that sees ONLY file metadata (paths, languages, token costs) —
no diff content is sent, keeping the grouping call cheap.

Each group is then capped at a per-group token budget and max file count
to ensure the downstream agent loop stays within its context window.

Grouping strategies:
  - ≤1 file:  trivial — one group, no logic needed
  - ≤SMALL_CHANGE_THRESHOLD: all files in one group (deterministic)
  - >threshold: LLM groups by semantic similarity, then split by budget
  - Fallback: one-file-per-group if LLM fails

References: new_architecture.md Phase 5, Alibaba OCR grouping.go
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import structlog

from src.agents.file_selection import FileDecision

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMALL_CHANGE_THRESHOLD = 5  # ≤ this many files → no LLM grouping needed
DEFAULT_MAX_FILES_PER_GROUP = 10
DEFAULT_TOKEN_LIMIT_PER_GROUP = 50_000


# ---------------------------------------------------------------------------
# FileGroup — the output of group_files()
# ---------------------------------------------------------------------------

@dataclass
class FileGroup:
    """A group of semantically related files to be reviewed together.

    Attributes:
        id: Unique group identifier.
        label: Human-readable label for the group (e.g. "Auth flow changes").
        files: List of file paths in this group.
        file_decisions: The FileDecision objects for each file.
        total_tokens: Sum of estimated diff tokens for all files.
        grouping_method: How this group was created — "trivial", "small",
            "llm_semantic", or "fallback".
    """
    id: int
    label: str
    files: list[str] = field(default_factory=list)
    file_decisions: list[FileDecision] = field(default_factory=list)
    total_tokens: int = 0
    grouping_method: str = "trivial"


# ---------------------------------------------------------------------------
# Deterministic grouping (small change sets)
# ---------------------------------------------------------------------------

def _group_trivial(decisions: list[FileDecision]) -> list[FileGroup]:
    """One group with all files — for ≤1 file or ≤SMALL_CHANGE_THRESHOLD."""
    if not decisions:
        return []

    total_tokens = sum(d.diff_tokens for d in decisions)
    files = [d.file.path for d in decisions]

    return [FileGroup(
        id=0,
        label="All changes",
        files=files,
        file_decisions=list(decisions),
        total_tokens=total_tokens,
        grouping_method="trivial" if len(decisions) <= 1 else "small",
    )]


# ---------------------------------------------------------------------------
# Directory-based grouping heuristic (no LLM fallback)
# ---------------------------------------------------------------------------

def _group_by_directory(decisions: list[FileDecision]) -> list[FileGroup]:
    """Group files by their top-level directory — deterministic fallback.

    Files in the same top-level directory are assumed to be related.
    Root-level files go into an "other" group.
    """
    dir_map: dict[str, list[FileDecision]] = {}

    for decision in decisions:
        path = decision.file.path
        parts = path.replace("\\", "/").split("/")
        # Use top-level directory, or "root" for root-level files
        top_dir = parts[0] if len(parts) > 1 else "_root"
        dir_map.setdefault(top_dir, []).append(decision)

    groups: list[FileGroup] = []
    for group_id, (dir_name, dir_decisions) in enumerate(
        sorted(dir_map.items())
    ):
        label = f"Changes in {dir_name}/" if dir_name != "_root" else "Root-level changes"
        total_tokens = sum(d.diff_tokens for d in dir_decisions)

        groups.append(FileGroup(
            id=group_id,
            label=label,
            files=[d.file.path for d in dir_decisions],
            file_decisions=dir_decisions,
            total_tokens=total_tokens,
            grouping_method="fallback",
        ))

    return groups


# ---------------------------------------------------------------------------
# LLM-based semantic grouping
# ---------------------------------------------------------------------------

def _build_grouping_prompt(decisions: list[FileDecision]) -> str:
    """Build the LLM prompt for semantic file grouping.

    We send ONLY file metadata — paths, languages, estimated tokens.
    No diff content! This keeps the grouping call cheap.
    """
    file_listing: list[str] = []
    for i, d in enumerate(decisions):
        lang = d.file.language or _infer_language(d.file.path)
        file_listing.append(
            f"  {i}: {d.file.path} (lang={lang}, "
            f"additions={d.file.additions}, deletions={d.file.deletions}, "
            f"tokens≈{d.diff_tokens})"
        )

    file_list_str = "\n".join(file_listing)

    return f"""Group the following changed files by semantic relationship for a code review.
Files that are likely related (same feature, same subsystem, caller/callee, etc.) should be in the same group.

Files:
{file_list_str}

Rules:
- Each group should have at most {DEFAULT_MAX_FILES_PER_GROUP} files
- Give each group a short descriptive label
- A file may only appear in ONE group
- Return valid JSON

Return a JSON object with this exact structure:
{{
  "groups": [
    {{
      "label": "Short description of what these files do together",
      "file_indices": [0, 3, 7]
    }},
    ...
  ]
}}

Respond ONLY with the JSON object, no other text."""


def _infer_language(path: str) -> str:
    """Infer language from file extension — lightweight helper."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".jsx": "javascript", ".go": "go",
        ".java": "java", ".rs": "rust", ".rb": "ruby", ".php": "php",
        ".cs": "csharp", ".cpp": "cpp", ".c": "c", ".h": "c",
        ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
        ".sql": "sql", ".sh": "shell",
        ".yaml": "config", ".yml": "config", ".json": "config",
        ".toml": "config", ".xml": "config",
    }
    lower = path.lower()
    dot_pos = lower.rfind(".")
    if dot_pos == -1:
        return "unknown"
    ext = lower[dot_pos:]
    return ext_map.get(ext, "unknown")


def _parse_grouping_response(
    response_text: str,
    decisions: list[FileDecision],
) -> list[FileGroup] | None:
    """Parse the LLM's grouping response into FileGroup objects.

    Returns None if parsing fails (caller should fallback).
    """
    # Extract JSON from the response (may be wrapped in markdown)
    json_text = response_text.strip()

    # Strip markdown code fences if present
    code_block_match = re.search(
        r'```(?:json)?\s*\n?(.*?)```', json_text, re.DOTALL
    )
    if code_block_match:
        json_text = code_block_match.group(1).strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        logger.warning("grouping.parse_failed", response_preview=json_text[:200])
        return None

    if not isinstance(data, dict) or "groups" not in data:
        logger.warning("grouping.missing_groups_key")
        return None

    raw_groups = data["groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        logger.warning("grouping.empty_groups")
        return None

    # Validate and build FileGroup objects
    used_indices: set[int] = set()
    groups: list[FileGroup] = []

    for group_id, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            continue

        label = raw_group.get("label", f"Group {group_id}")
        file_indices = raw_group.get("file_indices", [])

        if not isinstance(file_indices, list):
            continue

        # Filter valid, unused indices
        valid_indices = [
            i for i in file_indices
            if isinstance(i, int) and 0 <= i < len(decisions) and i not in used_indices
        ]

        if not valid_indices:
            continue

        used_indices.update(valid_indices)

        group_decisions = [decisions[i] for i in valid_indices]
        total_tokens = sum(d.diff_tokens for d in group_decisions)

        groups.append(FileGroup(
            id=group_id,
            label=str(label),
            files=[d.file.path for d in group_decisions],
            file_decisions=group_decisions,
            total_tokens=total_tokens,
            grouping_method="llm_semantic",
        ))

    # Handle any files not assigned to a group
    unassigned = [
        i for i in range(len(decisions)) if i not in used_indices
    ]
    if unassigned:
        ungrouped_decisions = [decisions[i] for i in unassigned]
        total_tokens = sum(d.diff_tokens for d in ungrouped_decisions)
        groups.append(FileGroup(
            id=len(groups),
            label="Other changes",
            files=[d.file.path for d in ungrouped_decisions],
            file_decisions=ungrouped_decisions,
            total_tokens=total_tokens,
            grouping_method="llm_semantic",
        ))

    return groups if groups else None


# ---------------------------------------------------------------------------
# Budget enforcement — split oversized groups
# ---------------------------------------------------------------------------

def _enforce_group_limits(
    groups: list[FileGroup],
    *,
    max_files_per_group: int = DEFAULT_MAX_FILES_PER_GROUP,
    token_limit: int = DEFAULT_TOKEN_LIMIT_PER_GROUP,
) -> list[FileGroup]:
    """Split groups that exceed per-group file count or token limits.

    This is a post-processing step that runs AFTER grouping (whether
    deterministic or LLM-based). It ensures no single group overloads
    the downstream agent loop.
    """
    result: list[FileGroup] = []
    next_id = 0

    for group in groups:
        if (
            len(group.files) <= max_files_per_group
            and group.total_tokens <= token_limit
        ):
            # Group is within limits
            result.append(FileGroup(
                id=next_id,
                label=group.label,
                files=group.files,
                file_decisions=group.file_decisions,
                total_tokens=group.total_tokens,
                grouping_method=group.grouping_method,
            ))
            next_id += 1
            continue

        # Group exceeds limits — split it
        # Sort by token cost (largest first) for better bin packing
        sorted_decisions = sorted(
            group.file_decisions,
            key=lambda d: d.diff_tokens,
            reverse=True,
        )

        current_files: list[str] = []
        current_decisions: list[FileDecision] = []
        current_tokens = 0
        split_num = 1

        for decision in sorted_decisions:
            # Check if adding this file would exceed limits
            if (
                current_files
                and (
                    len(current_files) >= max_files_per_group
                    or current_tokens + decision.diff_tokens > token_limit
                )
            ):
                # Flush current sub-group
                result.append(FileGroup(
                    id=next_id,
                    label=f"{group.label} (part {split_num})",
                    files=current_files,
                    file_decisions=current_decisions,
                    total_tokens=current_tokens,
                    grouping_method=group.grouping_method,
                ))
                next_id += 1
                split_num += 1
                current_files = []
                current_decisions = []
                current_tokens = 0

            current_files.append(decision.file.path)
            current_decisions.append(decision)
            current_tokens += decision.diff_tokens

        # Flush remaining
        if current_files:
            label = (
                f"{group.label} (part {split_num})"
                if split_num > 1
                else group.label
            )
            result.append(FileGroup(
                id=next_id,
                label=label,
                files=current_files,
                file_decisions=current_decisions,
                total_tokens=current_tokens,
                grouping_method=group.grouping_method,
            ))
            next_id += 1

    logger.info(
        "grouping.limits_enforced",
        input_groups=len(groups),
        output_groups=len(result),
        splits=len(result) - len(groups) if len(result) > len(groups) else 0,
    )

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def group_files(
    decisions: list[FileDecision],
    llm: object | None = None,
    *,
    max_files_per_group: int = DEFAULT_MAX_FILES_PER_GROUP,
    token_limit: int = DEFAULT_TOKEN_LIMIT_PER_GROUP,
) -> list[FileGroup]:
    """Group related files for review

    Grouping strategy:
      - ≤1 file: trivial grouping (no LLM, one group)
      - ≤SMALL_CHANGE_THRESHOLD files: all in one group (no LLM)
      - >threshold: call LLM with file metadata only (paths, no diffs)
        to produce semantic groups, then enforce token budget per group
      - Fallback: directory-based grouping if LLM fails

    After grouping, per-group limits (max files, token budget) are
    enforced. Oversized groups are split deterministically.

    Args:
        decisions: Selected FileDecision objects (only selected=True
            entries should be passed).
        llm: QuotaAwareFallbackLLM instance. If None, LLM grouping
            is skipped and deterministic heuristics are used.
        max_files_per_group: Maximum files in a single review group.
        token_limit: Maximum estimated diff tokens per group.

    Returns:
        List of FileGroup objects, each ready for a separate agent
        review loop call.
    """
    # Filter to only selected files
    selected = [d for d in decisions if d.selected]

    if not selected:
        return []

    # Trivial: 0-1 files
    if len(selected) <= 1:
        groups = _group_trivial(selected)
        logger.info(
            "grouping.trivial",
            files=len(selected),
            groups=len(groups),
        )
        return groups

    # Small change set: all in one group
    if len(selected) <= SMALL_CHANGE_THRESHOLD:
        groups = _group_trivial(selected)
        logger.info(
            "grouping.small_batch",
            files=len(selected),
            threshold=SMALL_CHANGE_THRESHOLD,
        )
        return _enforce_group_limits(
            groups,
            max_files_per_group=max_files_per_group,
            token_limit=token_limit,
        )

    # Large change set: try LLM semantic grouping
    if llm is not None:
        groups = await _llm_group_files(selected, llm)
        if groups is not None:
            logger.info(
                "grouping.llm_semantic",
                files=len(selected),
                groups=len(groups),
            )
            return _enforce_group_limits(
                groups,
                max_files_per_group=max_files_per_group,
                token_limit=token_limit,
            )

    # Fallback: directory-based grouping
    groups = _group_by_directory(selected)
    logger.info(
        "grouping.directory_fallback",
        files=len(selected),
        groups=len(groups),
    )
    return _enforce_group_limits(
        groups,
        max_files_per_group=max_files_per_group,
        token_limit=token_limit,
    )


async def _llm_group_files(
    decisions: list[FileDecision],
    llm: object,
) -> list[FileGroup] | None:
    """Call LLM to group files semantically.

    Sends ONLY file metadata (paths, languages, sizes) — NO diff content.
    This keeps the grouping call cheap (< 500 input tokens typically).

    Returns None on failure (caller should fallback to deterministic).
    """
    prompt = _build_grouping_prompt(decisions)

    messages = [
        {"role": "system", "content": (
            "You are a code review assistant. Group related files for "
            "efficient batch review. Return ONLY valid JSON."
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        # Use the LLM's invoke method (QuotaAwareFallbackLLM)
        response = await llm.invoke(  # type: ignore[union-attr]
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            timeout=15,
        )

        if response is None:
            logger.warning("grouping.llm_returned_none")
            return None

        groups = _parse_grouping_response(response.content, decisions)
        if groups is None:
            logger.warning("grouping.llm_parse_failed")
            return None

        return groups

    except Exception as e:
        logger.warning(
            "grouping.llm_error",
            error=str(e),
        )
        return None


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def format_grouping_summary(groups: list[FileGroup]) -> str:
    """Format a human-readable summary of file groups.

    Useful for logging and debugging.
    """
    lines: list[str] = [
        f"File Groups: {len(groups)} groups, "
        f"{sum(len(g.files) for g in groups)} files total"
    ]

    for g in groups:
        lines.append(
            f"  Group {g.id}: \"{g.label}\" — "
            f"{len(g.files)} files, ~{g.total_tokens} tokens "
            f"[{g.grouping_method}]"
        )
        for path in g.files[:5]:
            lines.append(f"    - {path}")
        if len(g.files) > 5:
            lines.append(f"    ... and {len(g.files) - 5} more")

    return "\n".join(lines)


def total_group_tokens(groups: list[FileGroup]) -> int:
    """Sum estimated tokens across all groups."""
    return sum(g.total_tokens for g in groups)
