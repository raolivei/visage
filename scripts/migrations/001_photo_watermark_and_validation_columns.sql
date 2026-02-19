-- Migration 001: Add watermark and validation columns to photos table
-- Idempotent: safe to run multiple times (skips if column already exists).
-- Fixes 500 on POST /api/packs/:id/photos when DB was created before these columns existed.

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN watermark_removed BOOLEAN DEFAULT FALSE;
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN original_s3_key TEXT;
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN watermark_job_id VARCHAR(36);
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN quality_score FLOAT;
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN is_valid VARCHAR(20) DEFAULT 'pending';
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN validation_errors JSONB DEFAULT '[]';
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN face_detected VARCHAR(20) DEFAULT 'pending';
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN face_bbox JSONB;
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN face_landmarks JSONB;
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;

DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN image_metadata JSONB DEFAULT '{}';
EXCEPTION
  WHEN duplicate_column THEN NULL;
END $$;
