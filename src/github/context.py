"""Request-scoped GitHub client for the review pipeline.

The LangGraph state is checkpointed to Postgres, so it must stay
serialisable — an open HTTP client can't live in it. Instead the worker
sets the client in this ContextVar before running the graph, and nodes
that need repository access (the agent loop's file_read / code_search
tools) read it from here.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

current_github_client: ContextVar[Any] = ContextVar("current_github_client", default=None)
