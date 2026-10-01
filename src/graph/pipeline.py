"""Main LangGraph pipeline — optimized single-pass review.

OPTIMIZED PIPELINE (Strategy 1 + 2 + Phase 1):
  File Selection → Triage → Reviewer → Critic → Summarizer

  - File Selection: Deterministic pure function, ZERO LLM tokens (filter files)
  - Triage:         Pure rules, ZERO LLM tokens (classifies PR tier)
  - Reviewer:       SINGLE LLM call (replaces 4 specialist agents)
  - Critic:         Rule-based only, ZERO LLM tokens (verify + dedup)
  - Summarizer:     Template-based, ZERO LLM tokens (format findings)

Total: 1 LLM call per review (down from 6-7).
Token savings: ~80% reduction.

Supports PostgresSaver for checkpointing (Supabase) and in-memory
MemorySaver for testing.

References: Token optimization §1 (single-pass), §2 (tiered review),
            new_architecture.md Phase 1 (file selection layer).
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from src.graph.large_pr import prepare_diff_for_review
from src.graph.nodes import (
    critic_node,
    file_selection_node,
    reviewer_node,
    summarizer_node,
    triage_node,
)
from src.llm.fallback import QuotaAwareFallbackLLM
from src.models.schemas import DiffContext, PRMetadata, ReviewState

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Node wrappers — inject LLM dependency via partial application
# ---------------------------------------------------------------------------

def _make_node(fn, llm: QuotaAwareFallbackLLM):
    """Create a LangGraph-compatible node function with LLM injected."""
    async def wrapper(state: ReviewState) -> dict:
        return await fn(state, llm)
    wrapper.__name__ = fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_review_graph(
    llm: QuotaAwareFallbackLLM,
    checkpointer: Any = None,
) -> StateGraph:
    """Build the optimized review pipeline graph.

    Pipeline: file_selection → triage → reviewer → critic → summarizer → END

    This is a simple linear graph — no fan-out, no conditional edges,
    no retry loops. Much simpler to debug and maintain.

    The file_selection node runs first (Phase 1) and filters the diff_context
    to only include files selected for review. This ensures the reviewer
    never sees excluded files (binary, generated, too-large, etc.).

    Args:
        llm: QuotaAwareFallbackLLM for the reviewer call.
        checkpointer: LangGraph checkpointer (PostgresSaver or MemorySaver).

    Returns:
        Compiled StateGraph ready to invoke.
    """
    graph = StateGraph(ReviewState)

    # Add nodes — each wraps a node function with the LLM injected
    graph.add_node("file_selection", _make_node(file_selection_node, llm))
    graph.add_node("triage", _make_node(triage_node, llm))
    graph.add_node("reviewer", _make_node(reviewer_node, llm))
    graph.add_node("critic", _make_node(critic_node, llm))
    graph.add_node("summarizer", _make_node(summarizer_node, llm))

    # Linear pipeline: file_selection → triage → reviewer → critic → summarizer → END
    graph.set_entry_point("file_selection")
    graph.add_edge("file_selection", "triage")
    graph.add_edge("triage", "reviewer")
    graph.add_edge("reviewer", "critic")
    graph.add_edge("critic", "summarizer")
    graph.add_edge("summarizer", END)

    # Compile with optional checkpointer
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# Pipeline runner — high-level entry point
# ---------------------------------------------------------------------------

async def run_review_pipeline(
    diff_context: DiffContext,
    pr_metadata: PRMetadata,
    llm: QuotaAwareFallbackLLM,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> dict:
    """Run the optimized review pipeline on a PR diff.

    This is the main entry point called by the worker. The pipeline:
    1. Triage — classify PR tier (zero tokens)
    2. Reviewer — single-pass LLM review (one call)
    3. Critic — rule-based verification (zero tokens)
    4. Summarizer — template-based formatting (zero tokens)

    Args:
        diff_context: Parsed and redacted diff context.
        pr_metadata: PR metadata from webhook.
        llm: QuotaAwareFallbackLLM instance.
        checkpointer: Optional LangGraph checkpointer.
        thread_id: Optional thread ID for checkpointing.

    Returns:
        Dict with keys: summary, verified_findings, degraded_mode,
        critic_available, failed_agents, large_pr_message, review_tier.
    """
    # Step 1: Large PR handling
    processed_diff, size_tier, large_pr_message = prepare_diff_for_review(diff_context)

    logger.info(
        "pipeline.starting",
        repo=pr_metadata.repo,
        pr=pr_metadata.pr_number,
        files=len(processed_diff.files),
        size_tier=size_tier,
    )

    # Step 2: Build initial state
    initial_state: ReviewState = {
        "diff_context": processed_diff,
        "file_decisions": [],
        "pr_metadata": pr_metadata,
        "active_agents": [],
        "findings": [],
        "verified_findings": [],
        "summary": "",
        "retry_count": 0,
        "failed_agents": [],
        "quota_skipped_agents": [],
        "degraded_mode": False,
        "critic_available": True,
        "model_usage": {},
        "review_tier": "",
        "triage_reason": "",
    }

    # Step 3: Build and run graph
    graph = build_review_graph(llm, checkpointer)

    config: dict[str, Any] = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    # Run the graph
    result = await graph.ainvoke(initial_state, config=config)

    # Step 4: Extract results
    summary = result.get("summary", "")
    verified_findings = result.get("verified_findings", [])
    failed_agents = result.get("failed_agents", [])
    critic_available = result.get("critic_available", True)
    review_tier = result.get("review_tier", "standard")
    degraded_mode = bool(failed_agents)

    # Prepend large PR message if applicable
    if large_pr_message:
        summary = f"> {large_pr_message}\n\n{summary}"

    logger.info(
        "pipeline.completed",
        repo=pr_metadata.repo,
        pr=pr_metadata.pr_number,
        review_tier=review_tier,
        verified_findings=len(verified_findings),
        failed_agents=failed_agents,
        critic_available=critic_available,
        degraded_mode=degraded_mode,
    )

    return {
        "summary": summary,
        "verified_findings": verified_findings,
        "degraded_mode": degraded_mode,
        "critic_available": critic_available,
        "failed_agents": failed_agents,
        "large_pr_message": large_pr_message,
        "review_tier": review_tier,
    }
