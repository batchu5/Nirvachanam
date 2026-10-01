"""Pydantic input/output schemas for every agent in the review pipeline.

Each agent has a typed *Input and *Output model so that the LangGraph nodes,
tests, and prompt-injection boundaries are all documented in one place.

Agents covered (Milestone 3):
  - Planner         (§2a) — routes files to specialist agents
  - BugAgent        (§1b) — detects logic errors, runtime bugs
  - SecurityAgent   (§1b) — detects vulnerabilities, secret leaks
  - StyleAgent      (§1b) — detects style / convention violations
  - TestAgent       (§1b) — flags missing test coverage
  - Critic          (§15c) — verifies, deduplicates, rejects bogus findings
  - Summarizer      (§2)  — produces markdown PR summary

Convention: every *Output model can be passed directly to
`model_validate_json(llm_response.content)` — i.e. the LLM is expected to
return JSON matching exactly this shape.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.enums import Category, Severity


# ═══════════════════════════════════════════════════════════════════════════
#  PLANNER (§2a)
# ═══════════════════════════════════════════════════════════════════════════

class FileRouting(BaseModel):
    """A single file → agent mapping produced by the planner."""
    file: str = Field(description="File path from the diff")
    agents: list[str] = Field(
        description=(
            "Which specialist agents should review this file. "
            "Valid values: bug_agent, security_agent, style_agent, test_agent"
        ),
    )
    reason: str = Field(
        default="",
        description="Brief reason for the routing decision",
    )


class PlannerInput(BaseModel):
    """Input contract for the planner node.

    Built by the graph orchestrator from ReviewState before calling the LLM.
    """
    files: list[FileInfo] = Field(
        default_factory=list,
        description="Metadata for each file in the diff",
    )
    pr_title: str = Field(default="", description="PR title for context")
    pr_languages: list[str] = Field(
        default_factory=list,
        description="Detected languages across all files in the diff",
    )
    total_additions: int = 0
    total_deletions: int = 0


class FileInfo(BaseModel):
    """Lightweight file metadata sent to the planner (not the full patch)."""
    path: str
    language: str | None = None
    is_new: bool = False
    is_deleted: bool = False
    additions: int = 0
    deletions: int = 0


# Fix the forward-reference in PlannerInput now that FileInfo is defined
PlannerInput.model_rebuild()


class PlannerOutput(BaseModel):
    """Structured output the planner LLM must return.

    The orchestrator falls back to extension-based routing (§2a) if this
    fails validation or comes back empty.
    """
    file_routings: list[FileRouting] = Field(
        default_factory=list,
        description="Per-file routing decisions",
    )
    skip_review: bool = Field(
        default=False,
        description="Set True if the PR has no reviewable code changes",
    )
    skip_reason: str = Field(
        default="",
        description="Why the PR was skipped (only when skip_review=True)",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SPECIALIST AGENTS — Shared base
# ═══════════════════════════════════════════════════════════════════════════

class AgentFinding(BaseModel):
    """A single finding produced by a specialist agent's LLM call.

    This is the *LLM-facing* schema — it does NOT include orchestrator-set
    fields like `content_signature` or `schema_version`.  Those are added
    by the graph node wrapper after validation.
    """
    file: str = Field(description="File path from the diff")
    line: int = Field(description="1-indexed line number in the NEW file")
    end_line: int | None = Field(
        default=None,
        description="End line for multi-line findings",
    )
    severity: Severity = Field(description="info | warning | critical")
    category: Category = Field(description="bug | security | style | test")
    message: str = Field(
        description="Clear explanation of the issue and why it matters",
    )
    suggested_fix: str | None = Field(
        default=None,
        description="Brief code snippet or description of how to fix it",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0.0–1.0 confidence this is a real issue",
    )
    language: str | None = Field(
        default=None,
        description="Detected language of the file",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  BUG AGENT
# ═══════════════════════════════════════════════════════════════════════════

class BugAgentInput(BaseModel):
    """Input contract for the bug detection agent."""
    diff_content: str = Field(
        description="XML-wrapped diff patches to analyze for bugs",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Languages present in the diff",
    )


class BugAgentOutput(BaseModel):
    """Structured output from the bug detection agent.

    Backward-compatible with the existing BugAgentOutput in schemas.py.
    """
    findings: list[AgentFinding] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  SECURITY AGENT (Gemini — §1b)
# ═══════════════════════════════════════════════════════════════════════════

class RedactionInfo(BaseModel):
    """Summary of a redacted secret passed to the security agent."""
    line: int
    type: str
    length: int


class SecurityAgentInput(BaseModel):
    """Input contract for the security agent.

    Includes redaction records so the agent can flag *where* secrets were
    found without ever seeing the actual secret values (§15a).
    """
    diff_content: str = Field(
        description="XML-wrapped diff patches (secrets already redacted)",
    )
    redaction_records: list[RedactionInfo] = Field(
        default_factory=list,
        description="Records of secrets redacted before LLM processing",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Languages present in the diff",
    )


class SecurityAgentOutput(BaseModel):
    """Structured output from the security agent."""
    findings: list[AgentFinding] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  STYLE AGENT (Groq — §1b)
# ═══════════════════════════════════════════════════════════════════════════

class StyleRule(BaseModel):
    """A custom style rule from .reviewrules.yaml."""
    language: str = Field(description="Language this rule applies to")
    rule_name: str = Field(description="e.g. max_line_length, docstring_required")
    rule_value: str = Field(description="e.g. '120', 'true', 'recommended'")


class StyleAgentInput(BaseModel):
    """Input contract for the style agent.

    Includes custom repo style rules from .reviewrules.yaml (§14) so the
    agent adapts to the project's conventions.
    """
    diff_content: str = Field(
        description="XML-wrapped diff patches to analyze for style issues",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Languages present in the diff",
    )
    custom_style_rules: list[StyleRule] = Field(
        default_factory=list,
        description="Custom style rules from .reviewrules.yaml",
    )
    custom_instructions: str = Field(
        default="",
        description="Freeform custom instructions from .reviewrules.yaml",
    )


class StyleAgentOutput(BaseModel):
    """Structured output from the style agent."""
    findings: list[AgentFinding] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  TEST AGENT (Groq — §1b)
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentInput(BaseModel):
    """Input contract for the test coverage agent."""
    diff_content: str = Field(
        description="XML-wrapped diff patches to analyze for test coverage",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Languages present in the diff",
    )
    test_files_in_pr: list[str] = Field(
        default_factory=list,
        description="Paths of test files already included in the PR",
    )


class TestAgentOutput(BaseModel):
    """Structured output from the test coverage agent."""
    findings: list[AgentFinding] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  CRITIC (Gemini — §15c, §11a)
# ═══════════════════════════════════════════════════════════════════════════

class CriticFindingVerdict(BaseModel):
    """The critic's verdict on a single finding."""
    finding_index: int = Field(
        description="0-based index into the input findings list",
    )
    verdict: str = Field(
        description="accept | reject | merge",
    )
    reason: str = Field(
        description="Why the finding was accepted, rejected, or merged",
    )
    merge_into: int | None = Field(
        default=None,
        description="If verdict=merge, the index of the finding to merge into",
    )
    adjusted_severity: Severity | None = Field(
        default=None,
        description="Override severity if the critic thinks it's wrong",
    )
    adjusted_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override confidence if the critic disagrees",
    )


class CriticInput(BaseModel):
    """Input contract for the critic verification node.

    The critic receives ALL findings from ALL specialist agents and the
    raw diff lines — it verifies each finding against the actual code.
    """
    findings: list[AgentFinding] = Field(
        description="All findings from specialist agents to verify",
    )
    diff_lines: list[str] = Field(
        default_factory=list,
        description="Raw diff lines for cross-referencing findings",
    )
    file_patches: dict[str, str] = Field(
        default_factory=dict,
        description="Per-file patch text for verification",
    )


class CriticOutput(BaseModel):
    """Structured output from the critic agent.

    The critic returns a verdict for every input finding.
    Findings with verdict='reject' are dropped before summarization.
    Findings with verdict='merge' are deduplicated.
    """
    verdicts: list[CriticFindingVerdict] = Field(default_factory=list)
    injection_echo_detected: bool = Field(
        default=False,
        description="True if any finding appeared to be a prompt injection echo",
    )
    notes: str = Field(
        default="",
        description="Optional notes about the verification process",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARIZER
# ═══════════════════════════════════════════════════════════════════════════

class SummarizerInput(BaseModel):
    """Input contract for the summarizer node."""
    findings: list[AgentFinding] = Field(
        description="Verified findings (post-critic) to summarize",
    )
    total_files: int = Field(
        default=0,
        description="Number of files in the PR",
    )
    total_additions: int = 0
    total_deletions: int = 0
    file_names: list[str] = Field(
        default_factory=list,
        description="Names of files reviewed",
    )
    degraded_agents: list[str] = Field(
        default_factory=list,
        description="Agents that were skipped or failed",
    )
    critic_available: bool = Field(
        default=True,
        description="Whether the LLM critic was used (vs rule-based fallback)",
    )


class SummarizerOutput(BaseModel):
    """Structured output from the summarizer agent.

    Backward-compatible with the existing SummaryOutput in schemas.py.
    """
    summary: str = Field(description="Markdown summary of the review")


# ═══════════════════════════════════════════════════════════════════════════
#  UNIFIED REVIEWER (Strategy 1 — single-pass mega-prompt)
# ═══════════════════════════════════════════════════════════════════════════

class ReviewerOutput(BaseModel):
    """Structured output from the unified reviewer agent.

    Replaces the separate BugAgentOutput, SecurityAgentOutput,
    StyleAgentOutput, and TestAgentOutput. The unified reviewer
    returns findings across ALL categories in a single response.
    """
    findings: list[AgentFinding] = Field(default_factory=list)
