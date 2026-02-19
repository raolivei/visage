"""
Database Configuration

SQLAlchemy async engine and session management.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from src.config import get_settings

settings = get_settings()

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncSession:
    """
    Dependency that yields database sessions.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Idempotent migration: ensure photos table has watermark/validation columns.
# Matches scripts/migrations/001_photo_watermark_and_validation_columns.sql
_PHOTOS_COLUMNS_MIGRATION = """
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN watermark_removed BOOLEAN DEFAULT FALSE;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN original_s3_key TEXT;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN watermark_job_id VARCHAR(36);
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN quality_score FLOAT;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN is_valid VARCHAR(20) DEFAULT 'pending';
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN validation_errors JSONB DEFAULT '[]';
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN face_detected VARCHAR(20) DEFAULT 'pending';
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN face_bbox JSONB;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN face_landmarks JSONB;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
DO $$
BEGIN
  ALTER TABLE photos ADD COLUMN image_metadata JSONB DEFAULT '{}';
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
"""


async def _ensure_photos_columns(conn):
    """Run idempotent migration to add any missing photos columns."""
    # Each statement is one "DO $$ ... END $$;" block (split strips the "END $$;" suffix)
    for block in _PHOTOS_COLUMNS_MIGRATION.strip().split("END $$;"):
        block = block.strip()
        if not block:
            continue
        await conn.execute(text(block + ";"))


async def init_db():
    """Initialize database tables and ensure photos columns exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_photos_columns(conn)


async def close_db():
    """Close database connections."""
    await engine.dispose()
