-- Migration 002: Add user rating columns to outputs table
-- Idempotent: safe to run multiple times (skips if column already exists).
-- Fixes 500 on GET /api/packs/:id/outputs when DB was created before these columns existed.

DO $$
BEGIN
  ALTER TABLE outputs ADD COLUMN user_rating BOOLEAN;
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE outputs ADD COLUMN rating_reason TEXT;
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE outputs ADD COLUMN rated_at TIMESTAMP WITHOUT TIME ZONE;
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;
