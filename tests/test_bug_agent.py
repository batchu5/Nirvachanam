"""Tests for the bug agent with mocked LLM responses."""

from __future__ import annotations

import json

import pytest

from src.agents.bug_agent import run_bug_agent, _parse_findings, _build_diff_content
from src.llm.providers import LLMResponse
from src.models.schemas import DiffContext, FileContext, HunkInfo

from tests.test_llm_providers import FakeProvider, FakeRedis, ProviderQuotaTracker
from src.llm.fallback import QuotaAwareFallbackLLM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_diff_context():
    """A DiffContext with one Python file containing a bug."""
    return DiffContext(
        files=[
            FileContext(
                path="src/auth/login.py",
                language="python",
                hunks=[
                    HunkInfo(
                        source_start=10,
                        source_length=6,
                        target_start=10,
                        target_length=8,
                        added_lines=[
                            (12, '    # BUG: SQL injection vulnerability'),
                            (13, "    query = f\"SELECT * FROM users WHERE username = '{username}'\""),
                        ],
                        removed_lines=[],
                    ),
                ],
                patch=(
                    "--- a/src/auth/login.py\n"
                    "+++ b/src/auth/login.py\n"
                    "@@ -10,6 +10,8 @@\n"
                    "     if not username:\n"
                    "         raise ValueError(\"Username required\")\n"
                    "+    # BUG: SQL injection vulnerability\n"
                    "+    query = f\"SELECT * FROM users WHERE username = '{username}'\"\n"
                    "     connection = get_db_connection()\n"
                ),
                additions=2,
                deletions=0,
            ),
        ],
        total_additions=2,
        total_deletions=0,
        base_sha="base123",
        head_sha="head456",
        raw_patch="...",
    )


@pytest.fixture
def mock_bug_response():
    """Simulated LLM response with a finding."""
    findings = {
        "findings": [
            {
                "file": "src/auth/login.py",
                "line": 13,
                "severity": "critical",
                "category": "bug",
                "message": "SQL injection vulnerability: user input is directly interpolated into SQL query string.",
                "suggested_fix": "Use parameterized query: cursor.execute('SELECT * FROM users WHERE username = %s', (username,))",
                "confidence": 0.9,
                "agent": "bug_agent",
                "language": "python",
            }
        ]
    }
    return LLMResponse(
        content=json.dumps(findings),
        tokens_used=150,
        input_tokens=100,
        output_tokens=50,
        model="gemini-2.5-flash",
        provider="gemini",
    )


@pytest.fixture
def mock_empty_response():
    """Simulated LLM response with no findings."""
    return LLMResponse(
        content=json.dumps({"findings": []}),
        tokens_used=80,
        input_tokens=70,
        output_tokens=10,
        model="gemini-2.5-flash",
        provider="gemini",
    )


@pytest.fixture
def make_llm(mock_bug_response):
    """Create a QuotaAwareFallbackLLM with a fake provider."""
    def _make(response=None, error=None):
        provider = FakeProvider(
            "gemini",
            response=response or mock_bug_response,
            error=error,
        )
        redis = FakeRedis()
        quota = ProviderQuotaTracker(redis)
        return QuotaAwareFallbackLLM(providers=[provider], quota=quota)
    return _make


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseFinding:
    """Tests for _parse_findings."""

    def test_parse_valid_findings(self):
        raw = json.dumps({
            "findings": [
                {
                    "file": "test.py",
                    "line": 10,
                    "severity": "warning",
                    "category": "bug",
                    "message": "potential issue",
                    "agent": "bug_agent",
                }
            ]
        })
        findings = _parse_findings(raw)
        assert len(findings) == 1
        assert findings[0].file == "test.py"
        assert findings[0].severity == "warning"
        assert findings[0].category == "bug"

    def test_parse_empty_findings(self):
        raw = json.dumps({"findings": []})
        findings = _parse_findings(raw)
        assert findings == []

    def test_parse_invalid_finding_skipped(self):
        """Invalid individual findings should be skipped, not crash."""
        raw = json.dumps({
            "findings": [
                {"file": "test.py", "line": 10, "severity": "warning",
                 "category": "bug", "message": "valid", "agent": "bug_agent"},
                {"file": "test.py"},  # Missing required fields
            ]
        })
        findings = _parse_findings(raw)
        assert len(findings) == 1  # Only the valid one

    def test_parse_sets_defaults(self):
        raw = json.dumps({
            "findings": [
                {"file": "test.py", "line": 5, "severity": "info",
                 "message": "note", "language": "python"}
            ]
        })
        findings = _parse_findings(raw)
        assert len(findings) == 1
        assert findings[0].category == "bug"  # Default
        assert findings[0].agent == "bug_agent"  # Default
        assert findings[0].confidence == 0.5  # Default


class TestBuildDiffContent:
    """Tests for _build_diff_content."""

    def test_builds_content_from_diff(self, sample_diff_context):
        content = _build_diff_content(sample_diff_context)
        assert "src/auth/login.py" in content
        assert "python" in content
        assert "SQL injection" in content

    def test_empty_diff_returns_empty(self):
        empty_ctx = DiffContext(files=[], total_additions=0, total_deletions=0)
        content = _build_diff_content(empty_ctx)
        assert content.strip() == ""


class TestRunBugAgent:
    """Tests for run_bug_agent."""

    async def test_successful_analysis(self, sample_diff_context, make_llm, mock_bug_response):
        llm = make_llm(response=mock_bug_response)
        findings = await run_bug_agent(sample_diff_context, llm)
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert findings[0].file == "src/auth/login.py"
        assert "SQL injection" in findings[0].message

    async def test_empty_diff_returns_empty(self, make_llm):
        empty_ctx = DiffContext(files=[], total_additions=0, total_deletions=0)
        llm = make_llm()
        findings = await run_bug_agent(empty_ctx, llm)
        assert findings == []

    async def test_no_findings(self, sample_diff_context, make_llm, mock_empty_response):
        llm = make_llm(response=mock_empty_response)
        findings = await run_bug_agent(sample_diff_context, llm)
        assert findings == []

    async def test_all_providers_fail_returns_empty(self, sample_diff_context):
        """When all LLM providers fail, bug_agent returns empty (graceful degradation)."""
        provider = FakeProvider("gemini", error=Exception("503 Service Unavailable"))
        redis = FakeRedis()
        quota = ProviderQuotaTracker(redis)
        llm = QuotaAwareFallbackLLM(providers=[provider], quota=quota)

        findings = await run_bug_agent(sample_diff_context, llm)
        assert findings == []

    async def test_invalid_json_triggers_repair(self, sample_diff_context):
        """When LLM returns invalid JSON, a repair retry should be attempted."""
        # First call returns garbage, second call returns valid JSON
        call_count = 0

        class RetryProvider:
            @property
            def name(self):
                return "gemini"

            async def invoke(self, messages, response_schema=None, temperature=0.2, max_tokens=2000):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return LLMResponse(
                        content="this is not json!!!",
                        tokens_used=50,
                        model="gemini",
                        provider="gemini",
                    )
                return LLMResponse(
                    content=json.dumps({"findings": []}),
                    tokens_used=50,
                    model="gemini",
                    provider="gemini",
                )

        redis = FakeRedis()
        quota = ProviderQuotaTracker(redis)
        llm = QuotaAwareFallbackLLM(providers=[RetryProvider()], quota=quota)

        findings = await run_bug_agent(sample_diff_context, llm)
        assert findings == []
        assert call_count == 2  # Original + repair retry
