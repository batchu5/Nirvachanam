"""Core data models for the review pipeline.

These are the shared contracts between webhook → queue → agents → GitHub posting.
All schemas follow the PRD §2 (ReviewState), §3 (Finding), §5 (DiffContext).
"""

from __future__ import annotations

import hashlib
import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from src.models.enums import Category, Severity


# ---------------------------------------------------------------------------
# §3 — Finding (shared contract between all agents)
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """A single review finding produced by a specialist agent.

    This is the core output schema — every agent returns list[Finding].
    The `content_signature` is set by the orchestrator AFTER the agent returns
    (using actual file content), not by the LLM itself.
    """
    schema_version: int = 1
    file: str
    line: int
    end_line: int | None = None
    severity: Severity
    category: Category
    message: str
    suggested_fix: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    agent: str
    language: str | None = None
    content_signature: str | None = None  # SHA-256 of ±2 lines context window


# ---------------------------------------------------------------------------
# §3 — content_signature computation
# ---------------------------------------------------------------------------

def compute_content_signature(
    file_lines: list[str], line: int, window: int = 2
) -> str:
    """Hash a ±2-line window around the flagged line for drift-safe matching.

    Args:
        file_lines: All lines of the file (0-indexed list).
        line: 1-indexed line number of the finding.
        window: Number of lines above/below to include.

    Returns:
        16-char hex prefix of SHA-256 hash.
    """
    start = max(0, line - 1 - window)
    end = min(len(file_lines), line + window)
    content = "\n".join(file_lines[start:end]).strip()
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Diff parsing models (§5)
# ---------------------------------------------------------------------------

class HunkInfo(BaseModel):
    """A single hunk from a unified diff."""
    source_start: int
    source_length: int
    target_start: int
    target_length: int
    added_lines: list[tuple[int, str]] = Field(
        default_factory=list, description="(line_number, content) for added lines"
    )
    removed_lines: list[tuple[int, str]] = Field(
        default_factory=list, description="(line_number, content) for removed lines"
    )


class FileContext(BaseModel):
    """A single file's diff context — parsed from unified diff."""
    path: str
    language: str | None = None
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    hunks: list[HunkInfo] = Field(default_factory=list)
    patch: str = ""  # Raw patch text for this file
    additions: int = 0
    deletions: int = 0


class RedactionRecord(BaseModel):
    """Record of a secret redacted from diff text before LLM processing."""
    line: int
    type: str  # e.g. "GitHub Token", "AWS Access Key"
    length: int


class DiffContext(BaseModel):
    """Full parsed diff for a PR — input to the LangGraph pipeline.

    Built by `build_diff_context()` which combines diff parsing + secret redaction.
    """
    files: list[FileContext] = Field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    base_sha: str = ""
    head_sha: str = ""
    raw_patch: str = ""  # Redacted patch text (secrets stripped)
    redaction_records: list[RedactionRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PR metadata (extracted from webhook payload)
# ---------------------------------------------------------------------------

class PRMetadata(BaseModel):
    """PR-level info extracted from the GitHub webhook payload."""
    repo: str            # "owner/repo"
    pr_number: int
    title: str = ""
    author: str = ""
    base_branch: str = ""
    head_branch: str = ""
    installation_id: int | None = None
    head_sha: str = ""
    base_sha: str = ""


class BugAgentOutput(BaseModel):
    """Structured output from the bug detection agent.

    The LLM returns this shape; we validate with model_validate_json()
    instead of manual json.loads() + dict wrangling.
    """
    findings: list[Finding] = Field(default_factory=list)


class SummaryOutput(BaseModel):
    """Structured output from the summarizer agent."""
    summary: str


class TokenUsage(BaseModel):
    """Per-agent token consumption — tracked for quota management."""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""


class ReviewState(TypedDict, total=False):
    """LangGraph graph state.

    TypedDict (not BaseModel) for LangGraph compatibility.
    `findings` uses Annotated[..., operator.add] for fan-in merging.
    """
    diff_context: DiffContext
    active_agents: list[str]
    findings: Annotated[list[Finding], operator.add]
    verified_findings: list[Finding]
    summary: str
    retry_count: int
    failed_agents: list[str]
    quota_skipped_agents: list[str]
    degraded_mode: bool
    critic_available: bool
    model_usage: dict[str, TokenUsage]
    pr_metadata: PRMetadata


class WebhookPullRequestPayload(BaseModel):
    """Partial model for GitHub `pull_request` webhook event.

    We only extract what's needed — GitHub payloads are huge.
    """
    action: str  # opened, synchronize, reopened, closed
    number: int  # PR number

    class PullRequest(BaseModel):
        title: str = ""
        head: dict  # {"sha": "...", "ref": "branch-name"}
        base: dict  # {"sha": "...", "ref": "main"}
        user: dict  # {"login": "username"}

    pull_request: PullRequest

    class Repository(BaseModel):
        full_name: str  # "owner/repo"

    repository: Repository

    class Installation(BaseModel):
        id: int

    installation: Installation | None = None

    def to_pr_metadata(self) -> PRMetadata:
        """Convert webhook payload to PRMetadata."""
        return PRMetadata(
            repo=self.repository.full_name,
            pr_number=self.number,
            title=self.pull_request.title,
            author=self.pull_request.user.get("login", ""),
            base_branch=self.pull_request.base.get("ref", ""),
            head_branch=self.pull_request.head.get("ref", ""),
            installation_id=self.installation.id if self.installation else None,
            head_sha=self.pull_request.head.get("sha", ""),
            base_sha=self.pull_request.base.get("sha", ""),
        )
