"""MCP server — exposes the PR review agent as tools for any MCP client.

Lets Claude Desktop, Claude Code, Cursor, etc. call the reviewer directly:

  * ``preview_review``       — zero-token plan: which files, which tier, est. cost
  * ``scan_secrets``         — zero-token secret scan (same redactor the pipeline uses)
  * ``review_diff``          — full LangGraph review of a raw unified diff
  * ``review_pull_request``  — fetch a GitHub PR, review it, optionally post the review

Run (stdio transport):
    python -m src.mcp_server

Stdio servers speak JSON-RPC over stdout, so all logging goes to stderr.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
)

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from src.agents.budget import estimate_review_cost, format_estimate_banner  # noqa: E402
from src.agents.file_selection import (  # noqa: E402
    ExcludeReason,
    select_files,
    selected_files,
    selection_summary,
)
from src.agents.triage import triage_from_decisions  # noqa: E402
from src.config import settings  # noqa: E402
from src.diff.parser import build_diff_context  # noqa: E402
from src.github.client import GitHubClient  # noqa: E402
from src.github.context import current_github_client  # noqa: E402
from src.github.reviewer import build_review_body, build_review_comments  # noqa: E402
from src.graph.pipeline import run_review_pipeline  # noqa: E402
from src.llm.fallback import QuotaAwareFallbackLLM  # noqa: E402
from src.llm.providers import GeminiProvider, GroqProvider  # noqa: E402
from src.llm.quota import ProviderQuotaTracker  # noqa: E402
from src.models.schemas import DiffContext, PRMetadata  # noqa: E402
from src.security.redaction import redact_secrets  # noqa: E402

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Local (Redis-free) LLM setup
# ---------------------------------------------------------------------------

class InMemoryQuotaTracker(ProviderQuotaTracker):
    """Quota tracker for single-user MCP sessions — no Redis needed.

    Only remembers providers that reported exhaustion, so the fallback chain
    still skips a rate-limited provider for the rest of its retry window.
    """

    def __init__(self) -> None:  # noqa: D107 — intentionally no Redis
        self._exhausted_until: dict[str, float] = {}

    async def can_call(self, provider: str) -> bool:
        return not await self.is_exhausted(provider)

    async def record_call(self, provider: str, tokens_used: int = 0) -> None:
        return None

    async def record_exhaustion(self, provider: str, retry_after: int = 60) -> None:
        import time

        self._exhausted_until[provider] = time.monotonic() + retry_after

    async def is_exhausted(self, provider: str) -> bool:
        import time

        return self._exhausted_until.get(provider, 0.0) > time.monotonic()


_llm: QuotaAwareFallbackLLM | None = None


def get_llm() -> QuotaAwareFallbackLLM:
    """Lazily build the fallback LLM (so zero-token tools work without keys)."""
    global _llm
    if _llm is None:
        providers = []
        if settings.gemini_api_key:
            providers.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model))
        if settings.groq_api_key:
            providers.append(GroqProvider(settings.groq_api_key, settings.groq_model))
        if not providers:
            raise RuntimeError(
                "No LLM provider configured: set GEMINI_API_KEY and/or GROQ_API_KEY in .env"
            )
        _llm = QuotaAwareFallbackLLM(providers=providers, quota=InMemoryQuotaTracker())
    return _llm


# ---------------------------------------------------------------------------
# Tool implementations (plain async functions — easy to unit test)
# ---------------------------------------------------------------------------

def _plan(diff_context: DiffContext) -> dict[str, Any]:
    decisions = select_files(diff_context)
    tier, reason = triage_from_decisions(decisions)
    estimate = estimate_review_cost(decisions)
    return {
        "tier": str(tier),
        "tier_reason": reason,
        "selected_files": [f.path for f in selected_files(decisions)],
        "excluded_files": {
            d.file.path: str(d.reason) for d in decisions if d.reason != ExcludeReason.NONE
        },
        "selection_summary": selection_summary(decisions),
        "estimate": format_estimate_banner(estimate),
    }


async def preview_review_impl(diff: str) -> dict[str, Any]:
    diff_context = build_diff_context(diff, PRMetadata(repo="local/diff", pr_number=0))
    plan = _plan(diff_context)
    plan["redacted_secrets"] = len(diff_context.redaction_records)
    return plan


async def scan_secrets_impl(text: str) -> dict[str, Any]:
    _, records = redact_secrets(text)
    return {
        "secrets_found": len(records),
        "findings": [r.model_dump() for r in records],
    }


def _format_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": result["summary"],
        "review_tier": result.get("review_tier"),
        "degraded_mode": result.get("degraded_mode", False),
        "failed_agents": result.get("failed_agents", []),
        "findings": [f.model_dump(mode="json") for f in result["verified_findings"]],
    }


async def review_diff_impl(diff: str, title: str = "") -> dict[str, Any]:
    metadata = PRMetadata(repo="local/diff", pr_number=0, title=title)
    diff_context = build_diff_context(diff, metadata)
    if not diff_context.files:
        return {"summary": "Nothing to review (all files were skipped).", "findings": []}
    result = await run_review_pipeline(diff_context, metadata, get_llm())
    return _format_result(result)


async def review_pull_request_impl(
    repo: str,
    pr_number: int,
    post_review: bool = False,
    installation_id: int | None = None,
    github: GitHubClient | None = None,
) -> dict[str, Any]:
    owns_client = github is None
    github = github or GitHubClient()
    try:
        pr = await github.get_pr(repo, pr_number, installation_id=installation_id)
        metadata = PRMetadata(
            repo=repo,
            pr_number=pr_number,
            title=pr.get("title", ""),
            author=(pr.get("user") or {}).get("login", ""),
            base_branch=pr["base"].get("ref", ""),
            head_branch=pr["head"].get("ref", ""),
            installation_id=installation_id,
            head_sha=pr["head"]["sha"],
            base_sha=pr["base"]["sha"],
        )
        raw = await github.get_pr_diff(repo, pr_number, installation_id=installation_id)
        diff_context = build_diff_context(raw, metadata)
        if not diff_context.files:
            return {"summary": "Nothing to review (all files were skipped).", "findings": []}

        token = current_github_client.set(github)
        try:
            result = await run_review_pipeline(diff_context, metadata, get_llm())
        finally:
            current_github_client.reset(token)

        output = _format_result(result)
        if post_review:
            if not installation_id:
                output["posted"] = False
                output["post_error"] = "installation_id is required to post as the GitHub App"
            else:
                comments = build_review_comments(result["verified_findings"], diff_context)
                body = build_review_body(
                    result["summary"],
                    result["verified_findings"],
                    degraded_agents=result["failed_agents"] or None,
                )
                await github.post_review(
                    repo,
                    pr_number,
                    metadata.head_sha,
                    body,
                    comments[: settings.max_comments_per_review],
                    installation_id=installation_id,
                )
                output["posted"] = True
        return output
    finally:
        if owns_client:
            await github.close()


# ---------------------------------------------------------------------------
# MCP server + tool registration
# ---------------------------------------------------------------------------

mcp = MCPServer(
    name="nirvachanam",
    title="Nirvachanam PR Reviewer",
    instructions=(
        "AI code reviewer. Use preview_review first for a free plan of what would be "
        "reviewed, then review_diff (local diff) or review_pull_request (GitHub PR). "
        "scan_secrets checks text for leaked credentials without calling an LLM."
    ),
)

_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


@mcp.tool(annotations=_READ_ONLY)
async def preview_review(diff: str) -> dict[str, Any]:
    """Plan a review of a unified diff without calling any LLM.

    Returns the files that would be reviewed or skipped (lockfiles, generated,
    binary, vendored), the review tier, and an estimated token cost.
    """
    return await preview_review_impl(diff)


@mcp.tool(annotations=_READ_ONLY)
async def scan_secrets(text: str) -> dict[str, Any]:
    """Scan text or a diff for leaked secrets (API keys, tokens, private keys).

    Returns the line and type of each secret found; never echoes the secret.
    """
    return await scan_secrets_impl(text)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def review_diff(diff: str, title: str = "") -> dict[str, Any]:
    """Run the full multi-agent review on a unified diff (e.g. `git diff` output).

    Returns a summary and verified findings with file, line, severity,
    category, message and suggested fix.
    """
    return await review_diff_impl(diff, title)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
async def review_pull_request(
    repo: str,
    pr_number: int,
    post_review: bool = False,
    installation_id: int | None = None,
) -> dict[str, Any]:
    """Review a GitHub pull request ("owner/repo", number).

    Public repos work without credentials. Set post_review=true (with the GitHub
    App installation_id) to also post the findings as an inline PR review.
    """
    return await review_pull_request_impl(repo, pr_number, post_review, installation_id)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
