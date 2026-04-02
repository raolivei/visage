"""
Database Configuration

SQLAlchemy async engine and session management.
"""

from sqlalchemy import inspect, text
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


async def _ensure_columns(conn):
    """Add any model columns missing from existing tables (idempotent).

    create_all() only creates new tables; it never adds columns to existing ones.
    This inspects every model table and issues ALTER TABLE ADD COLUMN for any column
    the DB doesn't have yet. Safe to run on every startup -- existing columns are
    skipped via duplicate_column exception handling.
    """
    db_tables = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_table_names()
    )
    for table in Base.metadata.sorted_tables:
        if table.name not in db_tables:
            continue
        db_cols = set(
            await conn.run_sync(
                lambda sync_conn, t=table.name: {
                    c["name"] for c in inspect(sync_conn).get_columns(t)
                }
            )
        )
        for col in table.columns:
            if col.name in db_cols:
                continue
            col_type = col.type.compile(dialect=conn.dialect)
            stmt = (
                f"DO $$ BEGIN "
                f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col_type}"
            )
            if col.default is not None and col.default.is_scalar:
                val = col.default.arg
                if isinstance(val, bool):
                    stmt += f" DEFAULT {'TRUE' if val else 'FALSE'}"
                elif isinstance(val, str):
                    stmt += f" DEFAULT '{val}'"
                elif isinstance(val, (int, float)):
                    stmt += f" DEFAULT {val}"
            stmt += "; EXCEPTION WHEN duplicate_column THEN NULL; END $$;"
            await conn.execute(text(stmt))


async def init_db():
    """Initialize database tables and ensure all model columns exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_columns(conn)


async def close_db():
    """Close database connections."""
    await engine.dispose()
