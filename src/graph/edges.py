"""Edge functions for the LangGraph review pipeline.

OPTIMIZED PIPELINE: The pipeline is now a simple linear chain
(triage → reviewer → critic → summarizer), so conditional edges
are no longer needed.

This module is kept for backwards compatibility and potential future
use if conditional routing is re-introduced.

References: Token optimization §1 (simplified pipeline).
"""

from __future__ import annotations

from typing import Literal

import structlog

from src.models.schemas import ReviewState

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Legacy edge functions (no longer wired into the optimized pipeline)
# Kept for backwards compatibility if needed.
# ---------------------------------------------------------------------------

def route_after_critic(
    state: ReviewState,
) -> Literal["summarizer"]:
    """After critic, always proceed to summarizer.

    In the optimized pipeline this is a simple edge, not conditional.
    Kept as a function for backwards compatibility.
    """
    return "summarizer"
