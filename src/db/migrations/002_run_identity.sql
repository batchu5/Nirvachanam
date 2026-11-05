-- ============================================================
-- Phase 7: Run Identity column + index
-- ============================================================
-- Adds a run_identity column to the reviews table for
-- identity-based dedup and checkpoint resume.
-- Idempotent: uses IF NOT EXISTS.

-- Add run_identity column to reviews table
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'reviews' AND column_name = 'run_identity'
    ) THEN
        ALTER TABLE reviews ADD COLUMN run_identity TEXT;
    END IF;
END $$;

-- Index for fast identity lookup
CREATE INDEX IF NOT EXISTS idx_reviews_run_identity
    ON reviews (repo, pr_number, run_identity)
    WHERE run_identity IS NOT NULL;
