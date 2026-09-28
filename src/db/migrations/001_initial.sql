-- ============================================================
-- AI PR Review Agent — Initial Schema (§13)
-- ============================================================
-- Idempotent: uses IF NOT EXISTS so re-running is safe.

-- Enable pgcrypto for gen_random_uuid() if not already enabled
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- -----------------------------------------------------------
-- Reviews — one row per (repo, pr, commit) review attempt
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo TEXT NOT NULL,
    pr_number INT NOT NULL,
    commit_sha TEXT NOT NULL,
    base_sha TEXT,                       -- for incremental diffs
    status TEXT DEFAULT 'pending',       -- pending, processing, completed, failed, degraded
    degraded_agents TEXT[],              -- which agents were skipped
    model_config JSONB,                  -- which models were used (gemini/groq per agent)
    token_usage JSONB,                   -- per-agent token counts
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE(repo, pr_number, commit_sha)  -- idempotency key
);

-- -----------------------------------------------------------
-- Findings — individual review comments from specialist agents
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS findings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID REFERENCES reviews(id) ON DELETE CASCADE,
    schema_version INT DEFAULT 1,
    file TEXT NOT NULL,
    line INT NOT NULL,
    end_line INT,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL,
    suggested_fix TEXT,
    confidence FLOAT DEFAULT 0.5,
    agent TEXT NOT NULL,
    language TEXT,
    content_signature TEXT,              -- SHA-256 hash of ±2 lines (§3, §6b)
    accepted BOOLEAN,                    -- NULL=pending, true=accepted, false=rejected
    github_comment_id BIGINT,           -- for auto-resolution tracking
    resolved BOOLEAN DEFAULT false,
    stale BOOLEAN DEFAULT false          -- true when invalidated by force push (§6c)
);

-- -----------------------------------------------------------
-- Eval examples — golden dataset for offline evaluation (§8)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_examples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    diff TEXT NOT NULL,
    expected_findings JSONB NOT NULL,
    source TEXT,                         -- 'manual', 'production_feedback'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------
-- Repo configs — cached .reviewrules.yaml per repository
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS repo_configs (
    repo TEXT PRIMARY KEY,
    config JSONB NOT NULL,               -- parsed .reviewrules.yaml
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------
-- Indexes for common queries
-- -----------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_reviews_repo_pr
    ON reviews (repo, pr_number);

CREATE INDEX IF NOT EXISTS idx_reviews_status
    ON reviews (status);

CREATE INDEX IF NOT EXISTS idx_findings_review_id
    ON findings (review_id);

CREATE INDEX IF NOT EXISTS idx_findings_file_line
    ON findings (file, line);

CREATE INDEX IF NOT EXISTS idx_findings_content_sig
    ON findings (content_signature)
    WHERE content_signature IS NOT NULL;
