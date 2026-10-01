"""Critic — rule-based verification, deduplication, and filtering of findings.

Now operates purely with deterministic rules (zero LLM tokens).
The old LLM-based critic functions have been removed as part of the
token optimization refactor (Strategy 1).

Core functions:
  - apply_rule_based_critic: Verify, dedup, and filter findings
  - is_injection_echo: Detect prompt-injection echo attacks

References: PRD §15c (injection-echo carve-outs), §11a (rule-based critic).
"""

from __future__ import annotations

from difflib import SequenceMatcher

import structlog

from src.models.schemas import DiffContext, Finding

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# §15c — Injection-echo detection with carve-outs
# ---------------------------------------------------------------------------

def _longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring between two strings."""
    matcher = SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    return s1[match.a: match.a + match.size]


def is_injection_echo(
    finding: Finding,
    diff_lines: list[str],
    similarity_threshold: float = 0.7,
    safe_verbatim_max_chars: int = 120,
) -> bool:
    """Check if a finding's message is suspiciously similar to diff content.

    Implements §15c carve-outs for legitimate quoting:
    - Short verbatim spans (< 120 chars) adjacent to referenced line are OK
    - Overlap inside suggested_fix field is OK (fixes quote the code being fixed)
    - Findings referencing [REDACTED:*] markers are always safe

    Args:
        finding: The finding to check.
        diff_lines: Raw diff lines for comparison.
        similarity_threshold: Max overlap ratio before flagging.
        safe_verbatim_max_chars: Max chars for legitimate quoting.

    Returns:
        True if the finding appears to be an injection echo.
    """
    # Carve-out 3: Findings referencing [REDACTED] markers are always safe
    if "[REDACTED:" in finding.message:
        return False

    for diff_line in diff_lines:
        if not diff_line.strip():
            continue

        overlap = _longest_common_substring(finding.message, diff_line)

        # Carve-out 1: Short overlap is fine — likely legitimate quoting
        if len(overlap) <= safe_verbatim_max_chars:
            continue

        # Carve-out 2: Overlap inside suggested_fix
        if finding.suggested_fix and overlap in finding.suggested_fix:
            continue

        # Long overlap NOT in a safe context → likely injection echo
        if len(overlap) / max(len(finding.message), 1) > similarity_threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Rule-based critic (deterministic — zero LLM tokens)
# ---------------------------------------------------------------------------

def _is_duplicate(finding: Finding, verified: list[Finding]) -> bool:
    """Check if a finding is a duplicate of an already-verified finding."""
    for existing in verified:
        if (
            existing.file == finding.file
            and existing.category == finding.category
            and abs(existing.line - finding.line) <= 3
        ):
            return True
    return False


def _line_exists_in_diff(file_path: str, line: int, diff_context: DiffContext) -> bool:
    """Verify that the referenced file and line actually exist in the diff."""
    for file_ctx in diff_context.files:
        if file_ctx.path == file_path:
            for hunk in file_ctx.hunks:
                for line_no, _ in hunk.added_lines:
                    if abs(line_no - line) <= 5:  # Allow some slack
                        return True
                # Also check if line is in the target range
                hunk_start = hunk.target_start
                hunk_end = hunk.target_start + hunk.target_length
                if hunk_start <= line <= hunk_end:
                    return True
            return True  # File exists but line check is loose
    return False


def apply_rule_based_critic(
    findings: list[Finding],
    diff_context: DiffContext,
) -> list[Finding]:
    """Deterministic critic — verifies, deduplicates, and filters findings.

    This is the PRIMARY critic in the optimized pipeline (zero LLM tokens).

    Checks applied:
    1. Drop low-confidence findings (< 0.7)
    2. Reject injection echoes (finding message suspiciously similar to diff)
    3. Verify referenced file/line exists in the diff
    4. Deduplicate findings within ±3 lines of the same category

    Args:
        findings: All findings from the unified reviewer.
        diff_context: Parsed diff for cross-referencing.

    Returns:
        Filtered list of findings that pass all rule-based checks.
    """
    # Collect raw diff lines for injection echo check
    diff_lines = []
    for file_ctx in diff_context.files:
        if file_ctx.patch:
            diff_lines.extend(file_ctx.patch.splitlines())

    verified: list[Finding] = []

    for finding in findings:
        # Drop low-confidence findings
        if finding.confidence < 0.7:
            logger.debug(
                "critic.rule_based.low_confidence",
                file=finding.file,
                line=finding.line,
                confidence=finding.confidence,
            )
            continue

        # Anti-injection: reject if finding message echoes diff content
        if is_injection_echo(finding, diff_lines, similarity_threshold=0.6):
            logger.warning(
                "critic.rule_based.injection_echo",
                file=finding.file,
                line=finding.line,
            )
            continue

        # Verify the referenced line exists in the diff
        if not _line_exists_in_diff(finding.file, finding.line, diff_context):
            logger.debug(
                "critic.rule_based.line_not_in_diff",
                file=finding.file,
                line=finding.line,
            )
            continue

        # Dedup
        if _is_duplicate(finding, verified):
            logger.debug(
                "critic.rule_based.duplicate",
                file=finding.file,
                line=finding.line,
            )
            continue

        verified.append(finding)

    logger.info(
        "critic.rule_based.completed",
        input_count=len(findings),
        output_count=len(verified),
        dropped=len(findings) - len(verified),
    )

    return verified
