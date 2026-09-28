"""GitHub review builder — maps findings to review API format.

The GitHub Pull Request Review API uses "position" (line offset within the
diff hunk), NOT file line numbers. This module handles the mapping from our
Finding objects (which use file line numbers) to the correct diff positions.

References: PRD §5 (line number mapping), §7a (error UX).
"""

from __future__ import annotations

from typing import Any

import structlog

from src.models.schemas import DiffContext, FileContext, Finding

logger = structlog.get_logger()


def _build_position_map(diff_context: DiffContext) -> dict[str, dict[int, int]]:
    """Build a mapping from (file, target_line_number) → diff position.

    GitHub's review API uses "position" which is the line offset within the
    entire diff for that file (1-indexed, counting all lines including
    context, additions, and deletions).

    Args:
        diff_context: Parsed diff context.

    Returns:
        Dict mapping file path → {target_line_number: position}.
    """
    position_map: dict[str, dict[int, int]] = {}

    for file_ctx in diff_context.files:
        file_positions: dict[int, int] = {}
        position = 0  # Cumulative position counter across hunks

        for hunk in file_ctx.hunks:
            # The hunk header (@@ ... @@) counts as position 1 for the first
            # hunk, and increments for subsequent hunks.
            position += 1  # Hunk header line

            # Walk through the hunk's lines by examining the patch text
            # We need to count all lines (context + added + removed)
            # and map target (new file) line numbers to positions.
            #
            # Since we have the structured hunk info, we can reconstruct this:
            # Each hunk covers source_start..source_start+source_length (old file)
            # and target_start..target_start+target_length (new file)
            #
            # For the position map, we need to count every line in the hunk
            # (context + additions + removals) and record the position for
            # each target line (context + additions).

            # Build sets for quick lookup
            added_lines = {line_no for line_no, _ in hunk.added_lines}
            removed_lines = {line_no for line_no, _ in hunk.removed_lines}

            # Walk through the hunk line by line
            src_line = hunk.source_start
            tgt_line = hunk.target_start

            # Total lines in hunk = source_length + added (or target_length + removed)
            # We iterate through all lines in order
            total_hunk_lines = hunk.source_length + len(hunk.added_lines)

            src_end = hunk.source_start + hunk.source_length
            tgt_end = hunk.target_start + hunk.target_length

            while src_line < src_end or tgt_line < tgt_end:
                position += 1

                if tgt_line in added_lines:
                    # This is an added line — exists in target only
                    file_positions[tgt_line] = position
                    tgt_line += 1
                elif src_line in removed_lines:
                    # This is a removed line — exists in source only
                    src_line += 1
                else:
                    # Context line — exists in both
                    file_positions[tgt_line] = position
                    src_line += 1
                    tgt_line += 1

                # Safety valve to prevent infinite loops
                if position > 10000:
                    logger.warning(
                        "reviewer.position_overflow",
                        file=file_ctx.path,
                    )
                    break

        position_map[file_ctx.path] = file_positions

    return position_map


def build_review_comments(
    findings: list[Finding],
    diff_context: DiffContext,
) -> list[dict[str, Any]]:
    """Convert findings to GitHub review API comment format.

    Maps each finding's file + line number to the correct diff position.
    Findings that can't be mapped (line not in diff) are logged and skipped.

    Args:
        findings: List of verified findings from the pipeline.
        diff_context: Parsed diff context (for position mapping).

    Returns:
        List of comment dicts ready for the GitHub Review API.
    """
    position_map = _build_position_map(diff_context)
    comments: list[dict[str, Any]] = []
    unmapped_count = 0

    for finding in findings:
        file_positions = position_map.get(finding.file)
        if not file_positions:
            logger.debug(
                "reviewer.file_not_in_diff",
                file=finding.file,
                line=finding.line,
            )
            unmapped_count += 1
            continue

        position = file_positions.get(finding.line)
        if position is None:
            # Try nearby lines (±3) in case of slight offset
            for offset in range(1, 4):
                position = file_positions.get(finding.line + offset)
                if position:
                    break
                position = file_positions.get(finding.line - offset)
                if position:
                    break

        if position is None:
            logger.debug(
                "reviewer.line_not_in_diff",
                file=finding.file,
                line=finding.line,
            )
            unmapped_count += 1
            continue

        # Build the comment body
        severity_emoji = {
            "critical": "🔴",
            "warning": "⚠️",
            "info": "ℹ️",
        }
        emoji = severity_emoji.get(finding.severity, "💡")

        body_parts = [
            f"{emoji} **{finding.severity.upper()}** ({finding.category})",
            "",
            finding.message,
        ]

        if finding.suggested_fix:
            body_parts.extend([
                "",
                "**Suggested fix:**",
                f"```\n{finding.suggested_fix}\n```",
            ])

        body_parts.extend([
            "",
            f"*Confidence: {finding.confidence:.0%} · Agent: {finding.agent}*",
        ])

        comment: dict[str, Any] = {
            "path": finding.file,
            "position": position,
            "body": "\n".join(body_parts),
        }
        comments.append(comment)

    if unmapped_count:
        logger.warning(
            "reviewer.unmapped_findings",
            count=unmapped_count,
            total=len(findings),
        )

    return comments


def build_review_body(
    summary: str,
    findings: list[Finding],
    degraded_agents: list[str] | None = None,
) -> str:
    """Build the review body markdown.

    Args:
        summary: Generated review summary from summarizer agent.
        findings: All findings (for stats).
        degraded_agents: List of agents that failed/were skipped.

    Returns:
        Markdown string for the review body.
    """
    parts: list[str] = []

    # Degraded mode banner
    if degraded_agents:
        agents_str = ", ".join(f"`{a}`" for a in degraded_agents)
        parts.append(
            f"> ⚠️ **Review completed in degraded mode** — {agents_str} "
            f"timed out or were skipped. Results may be incomplete.\n"
        )

    # Main summary
    parts.append(summary)

    # Stats footer
    if findings:
        severity_counts = {}
        for f in findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        stats_parts = []
        for sev in ["critical", "warning", "info"]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                stats_parts.append(f"{count} {sev}")

        if stats_parts:
            parts.append(f"\n---\n📊 **{len(findings)} findings:** {' · '.join(stats_parts)}")

    parts.append(
        "\n<sub>🤖 Generated by AI PR Review Agent</sub>"
    )

    return "\n".join(parts)
