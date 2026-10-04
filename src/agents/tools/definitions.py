"""Tool definitions for the agentic review loop.

Phase 3 of the OCR architecture migration. Defines the tools available
to the LLM agent during code review:

  1. CodeCommentTool — post a finding with file, line, code context
  2. FileReadTool — read full file content beyond the diff
  3. FileReadDiffTool — read another file's diff for cross-file context
  4. CodeSearchTool — grep the codebase for symbol usage
  5. TaskDoneTool — signal that the review is complete

Each tool has:
  - A JSON schema for the LLM to call with
  - An execute() method that returns string results
  - An optional repair hint for malformed arguments

References: new_architecture.md Phase 3, Alibaba OCR tools
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.models.schemas import DiffContext, FileContext

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Raw comment — intermediate representation from tool calls
# ---------------------------------------------------------------------------

@dataclass
class RawComment:
    """A review comment produced by the agent via the code_comment tool.

    This is the raw output before line resolution and validation.
    """
    file: str
    existing_code: str  # verbatim code snippet the comment refers to
    message: str
    severity: str = "warning"      # info | warning | critical
    category: str = "bug"          # bug | security | style | test
    suggested_fix: str | None = None
    confidence: float = 0.7
    language: str | None = None


# ---------------------------------------------------------------------------
# Tool protocol
# ---------------------------------------------------------------------------

class ReviewTool(ABC):
    """Base class for review tools available to the LLM agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in LLM function calling."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the tool's parameters."""
        ...

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool and return a string result."""
        ...

    @property
    def schema(self) -> dict[str, Any]:
        """Full OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Tool context — shared state available to all tools during a review
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Shared context available to all tools during a review session.

    Provides access to the diff, file content, and repository info
    that tools need to operate.

    Attributes:
        diff_context: The full parsed diff for the PR.
        file_patches: Map of file path → patch text for quick lookup.
        repo: Repository full name ("owner/repo").
        head_sha: HEAD commit SHA (for file reads).
        github_client: Optional GitHub client for file reads and search.
        installation_id: GitHub App installation ID.
        comments: Accumulated comments from code_comment tool calls.
    """
    diff_context: DiffContext
    file_patches: dict[str, str] = field(default_factory=dict)
    repo: str = ""
    head_sha: str = ""
    github_client: Any = None  # GitHubClient, optional to avoid circular imports
    installation_id: int | None = None
    comments: list[RawComment] = field(default_factory=list)

    def __post_init__(self):
        """Build the file_patches lookup from diff_context."""
        if not self.file_patches and self.diff_context:
            self.file_patches = {
                f.path: f.patch
                for f in self.diff_context.files
                if f.patch
            }


# ---------------------------------------------------------------------------
# 1. CodeCommentTool — post a review finding
# ---------------------------------------------------------------------------

class CodeCommentTool(ReviewTool):
    """Post a code review comment with file, code context, and explanation.

    The LLM uses this tool to report findings. Instead of line numbers,
    it provides the `existing_code` snippet — a verbatim quote from the
    diff. The line number is resolved deterministically later by the
    comment resolver (Phase 4).
    """

    def __init__(self, context: ToolContext):
        self._context = context

    @property
    def name(self) -> str:
        return "code_comment"

    @property
    def description(self) -> str:
        return (
            "Post a code review comment. Provide the file path, "
            "a verbatim code snippet from the diff that you're commenting on, "
            "and your review message. The line number will be resolved automatically."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "File path from the diff",
                },
                "existing_code": {
                    "type": "string",
                    "description": (
                        "Verbatim code snippet from the diff that this comment "
                        "refers to. Copy the exact lines from the diff."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": "Clear explanation of the issue and why it matters",
                },
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "critical"],
                    "description": "Severity level",
                },
                "category": {
                    "type": "string",
                    "enum": ["bug", "security", "style", "test"],
                    "description": "Review category",
                },
                "suggested_fix": {
                    "type": "string",
                    "description": "Brief code snippet or description of how to fix it",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "0.0–1.0 confidence this is a real issue",
                },
            },
            "required": ["file", "existing_code", "message", "severity", "category"],
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Record a comment and return confirmation."""
        comment = RawComment(
            file=arguments["file"],
            existing_code=arguments["existing_code"],
            message=arguments["message"],
            severity=arguments.get("severity", "warning"),
            category=arguments.get("category", "bug"),
            suggested_fix=arguments.get("suggested_fix"),
            confidence=arguments.get("confidence", 0.7),
            language=arguments.get("language"),
        )
        self._context.comments.append(comment)
        logger.debug(
            "tool.code_comment",
            file=comment.file,
            severity=comment.severity,
            category=comment.category,
        )
        return f"Comment recorded for {comment.file}: [{comment.severity}] {comment.category}"


# ---------------------------------------------------------------------------
# 2. FileReadTool — read full file content
# ---------------------------------------------------------------------------

class FileReadTool(ReviewTool):
    """Read the full content of a file from the repository.

    Allows the agent to see beyond the diff hunks — useful for
    understanding surrounding context, imports, class definitions, etc.
    """

    def __init__(self, context: ToolContext):
        self._context = context

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read the full content of a file from the repository at the "
            "current HEAD commit. Use this when you need more context "
            "beyond what's shown in the diff hunks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to repository root",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional: start line number (1-indexed) to read from",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional: end line number (1-indexed) to read to",
                },
            },
            "required": ["path"],
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Read file content via GitHub API or fallback to diff."""
        path = arguments["path"]
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")

        # Try GitHub API first
        if self._context.github_client and self._context.repo:
            try:
                content = await self._context.github_client.get_file_content(
                    repo=self._context.repo,
                    path=path,
                    ref=self._context.head_sha or "HEAD",
                    installation_id=self._context.installation_id,
                )
                if content is not None:
                    lines = content.splitlines()
                    if start_line and end_line:
                        lines = lines[max(0, start_line - 1):end_line]
                    elif start_line:
                        lines = lines[max(0, start_line - 1):]
                    return "\n".join(lines)
            except Exception as e:
                logger.warning("tool.file_read.api_error", path=path, error=str(e))

        # Fallback: extract from diff patch
        patch = self._context.file_patches.get(path)
        if patch:
            return f"[From diff patch — full file not available]\n{patch}"

        return f"File not found: {path}"


# ---------------------------------------------------------------------------
# 3. FileReadDiffTool — read another file's diff
# ---------------------------------------------------------------------------

class FileReadDiffTool(ReviewTool):
    """Read the diff of another file in the same PR.

    Enables cross-file review — the agent can check how changes
    in one file relate to changes in another.
    """

    def __init__(self, context: ToolContext):
        self._context = context

    @property
    def name(self) -> str:
        return "file_read_diff"

    @property
    def description(self) -> str:
        return (
            "Read the diff (patch) of another file in this pull request. "
            "Use this when you need to understand cross-file relationships, "
            "e.g., if a function signature changed in one file, check its "
            "callers in other files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read the diff for",
                },
            },
            "required": ["path"],
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Return the diff patch for a specified file."""
        path = arguments["path"]
        patch = self._context.file_patches.get(path)
        if patch:
            return f"Diff for {path}:\n{patch}"

        # Try fuzzy match
        for file_path, file_patch in self._context.file_patches.items():
            if file_path.endswith(path) or path.endswith(file_path):
                return f"Diff for {file_path} (fuzzy match):\n{file_patch}"

        available = list(self._context.file_patches.keys())
        return (
            f"No diff found for {path}. "
            f"Available files: {', '.join(available[:20])}"
        )


# ---------------------------------------------------------------------------
# 4. CodeSearchTool — grep the codebase for symbol usage
# ---------------------------------------------------------------------------

class CodeSearchTool(ReviewTool):
    """Search the codebase for a symbol, pattern, or string.

    Uses GitHub's code search API to find usages of functions,
    variables, or patterns across the repository.
    """

    def __init__(self, context: ToolContext):
        self._context = context

    @property
    def name(self) -> str:
        return "code_search"

    @property
    def description(self) -> str:
        return (
            "Search the codebase for a symbol, function name, or pattern. "
            "Use this to find all usages of a function that was modified, "
            "check if a deleted function is still referenced elsewhere, etc."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — function name, variable, or pattern",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional: restrict search to files matching this pattern (e.g., '*.py')",
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Search via GitHub API or fallback to diff grep."""
        query = arguments["query"]
        file_pattern = arguments.get("file_pattern", "")

        # Try GitHub code search API
        if self._context.github_client and self._context.repo:
            try:
                search_query = f"{query} repo:{self._context.repo}"
                if file_pattern:
                    search_query += f" path:{file_pattern}"

                response = await self._context.github_client._http.get(
                    "/search/code",
                    params={"q": search_query, "per_page": 10},
                    headers=self._context.github_client._get_auth_headers(
                        await self._context.github_client._get_installation_token(
                            self._context.installation_id
                        )
                    ) if self._context.installation_id else {},
                )
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    if items:
                        results = []
                        for item in items[:10]:
                            results.append(f"  - {item['path']} ({item.get('name', '')})")
                        return f"Found {data.get('total_count', 0)} results for '{query}':\n" + "\n".join(results)
                    return f"No results found for '{query}'"
            except Exception as e:
                logger.warning("tool.code_search.api_error", query=query, error=str(e))

        # Fallback: search within the diff patches
        matches: list[str] = []
        for path, patch in self._context.file_patches.items():
            if file_pattern and not fnmatch_match(path, file_pattern):
                continue
            for i, line in enumerate(patch.splitlines(), 1):
                if query.lower() in line.lower():
                    matches.append(f"  {path}:{i}: {line.strip()}")

        if matches:
            return f"Found {len(matches)} matches for '{query}' in PR diffs:\n" + "\n".join(matches[:20])
        return f"No matches found for '{query}' in the PR diffs."


# ---------------------------------------------------------------------------
# 5. TaskDoneTool — signal completion
# ---------------------------------------------------------------------------

class TaskDoneTool(ReviewTool):
    """Signal that the review is complete.

    The agent calls this when it has finished reviewing all files
    in the group. The agent loop terminates when this tool is invoked.
    """

    def __init__(self, context: ToolContext):
        self._context = context

    @property
    def name(self) -> str:
        return "task_done"

    @property
    def description(self) -> str:
        return (
            "Signal that you have completed your review of all files. "
            "Call this when you have examined all the code and posted "
            "all relevant comments via code_comment."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what you reviewed and key findings",
                },
            },
            "required": ["summary"],
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Mark review as complete."""
        summary = arguments.get("summary", "Review complete")
        logger.info(
            "tool.task_done",
            comments_posted=len(self._context.comments),
            summary=summary,
        )
        return f"Review complete. {len(self._context.comments)} comments posted."


# ---------------------------------------------------------------------------
# Helper: build all tools for a review session
# ---------------------------------------------------------------------------

def build_review_tools(context: ToolContext) -> list[ReviewTool]:
    """Create all review tools with shared context.

    Args:
        context: Shared ToolContext for the review session.

    Returns:
        List of all available ReviewTool instances.
    """
    return [
        CodeCommentTool(context),
        FileReadTool(context),
        FileReadDiffTool(context),
        CodeSearchTool(context),
        TaskDoneTool(context),
    ]


def find_tool(name: str, tools: list[ReviewTool]) -> ReviewTool | None:
    """Find a tool by name."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None


# ---------------------------------------------------------------------------
# fnmatch helper (avoid import at module level for code_search fallback)
# ---------------------------------------------------------------------------

def fnmatch_match(path: str, pattern: str) -> bool:
    """Simple glob matching for code search file filtering."""
    import fnmatch
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path.lower(), pattern)
