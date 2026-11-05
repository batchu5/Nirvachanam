"""Regression tests for LangGraph state wiring."""

from __future__ import annotations

import typing
from unittest.mock import AsyncMock

import pytest

from src.agents.reviewer import run_reviewer
from src.graph.nodes import _generate_summary
from src.llm.fallback import AllProvidersExhaustedError
from src.models.schemas import DiffContext, ReviewState


def test_rule_groups_is_part_of_graph_state():
    # LangGraph silently drops keys that aren't in the state schema. Missing
    # `rule_groups` meant the agent loop never ran (always one-shot fallback).
    assert "rule_groups" in typing.get_type_hints(ReviewState)


def test_incomplete_review_never_claims_no_issues():
    summary = _generate_summary([], DiffContext(files=[]), "standard", incomplete=True)
    assert "No issues detected" not in summary
    assert "could not be completed" in summary


async def test_reviewer_raises_when_no_provider_answers(sample_diff_context=None):
    from src.diff.parser import build_diff_context
    from src.models.schemas import PRMetadata
    from tests.conftest import SAMPLE_DIFF

    ctx = build_diff_context(SAMPLE_DIFF, PRMetadata(repo="o/r", pr_number=1))
    llm = AsyncMock()
    llm.invoke.return_value = None
    with pytest.raises(AllProvidersExhaustedError):
        await run_reviewer(ctx, llm)


def test_tool_calls_with_braces_in_code_are_not_dropped():
    # Regression: a `[^{}]*` regex silently dropped this SQL-injection comment
    # in a live review because the quoted f-string contains braces.
    from src.agents.agent_loop import _extract_tool_calls_from_response

    text = """I'll record the issues.
```json
{"name": "code_comment", "arguments": {"file": "app/db.py",
  "existing_code": "f\\"SELECT * FROM users WHERE username LIKE '%{term}%'\\"",
  "message": "SQL injection", "severity": "critical", "category": "security"}}
```
```json
{"name": "code_comment", "arguments": {"file": "web/app.js",
  "existing_code": "if (user) { return {id: user.id}; }",
  "message": "leaks id", "severity": "warning", "category": "security"}}
```
{"name": "task_done", "arguments": {"summary": "done"}}
"""
    calls = _extract_tool_calls_from_response(text)
    assert [c["name"] for c in calls] == ["code_comment", "code_comment", "task_done"]
    assert "{term}" in calls[0]["arguments"]["existing_code"]
    assert calls[1]["arguments"]["file"] == "web/app.js"


def test_openai_style_tool_calls_with_string_arguments():
    from src.agents.agent_loop import _extract_tool_calls_from_response

    text = (
        '{"tool_calls": [{"function": {"name": "file_read", '
        '"arguments": "{\\"path\\": \\"a.py\\"}"}}]}'
    )
    calls = _extract_tool_calls_from_response(text)
    assert calls == [{"name": "file_read", "arguments": {"path": "a.py"}}]


async def test_empty_reply_falls_back_to_next_provider():
    from src.llm.fallback import QuotaAwareFallbackLLM
    from src.llm.providers import LLMResponse

    class P:
        def __init__(self, name, content):
            self.name, self.content, self.calls = name, content, 0

        async def invoke(self, **kw):
            self.calls += 1
            return LLMResponse(content=self.content, tokens_used=1, input_tokens=1,
                               output_tokens=0, model="m", provider=self.name)

    quota = AsyncMock()
    quota.is_exhausted.return_value = False
    quota.can_call.return_value = True
    empty, good = P("groq", ""), P("gemini", "ok")
    llm = QuotaAwareFallbackLLM([empty, good], quota, retry_backoff_seconds=0)
    resp = await llm.invoke([{"role": "user", "content": "hi"}])
    assert resp.content == "ok" and empty.calls == 1


async def test_transient_failure_is_retried_on_second_pass():
    from src.llm.fallback import QuotaAwareFallbackLLM
    from src.llm.providers import LLMResponse

    class Flaky:
        name = "gemini"
        calls = 0

        async def invoke(self, **kw):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("503 UNAVAILABLE")
            return LLMResponse(content="ok", tokens_used=1, input_tokens=1,
                               output_tokens=0, model="m", provider="gemini")

    quota = AsyncMock()
    quota.is_exhausted.return_value = False
    quota.can_call.return_value = True
    p = Flaky()
    llm = QuotaAwareFallbackLLM([p], quota, retry_backoff_seconds=0)
    assert (await llm.invoke([{"role": "user", "content": "hi"}])).content == "ok"
    assert p.calls == 2
