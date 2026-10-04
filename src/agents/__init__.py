"""Agent module — optimized single-pass code review pipeline.

OPTIMIZED AGENTS (Strategy 1 + 2 + Phase 1-6):
  - file_selection: Deterministic file selection (zero LLM tokens, pure function)
  - triage: Rule-based PR classification (zero LLM tokens)
  - reviewer: Unified single-pass reviewer (one LLM call)
  - critic: Rule-based verification and deduplication (zero LLM tokens)
  - grouping: Semantic file grouping — LLM or deterministic (Phase 5)
  - budget: Token budget & cost estimation — pre-run projection (Phase 6)

LEGACY AGENTS (kept for backwards compatibility, no longer in pipeline):
  - planner: Routes files to specialist agents (§2a)
  - bug_agent: Detects logic errors and runtime bugs (§1b)
  - security_agent: Detects vulnerabilities (§1b)
  - style_agent: Detects style violations (§1b)
  - test_agent: Flags missing test coverage (§1b)
  - summarizer: Produces PR review summary (§2)
"""
