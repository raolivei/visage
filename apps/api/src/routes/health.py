"""
Health Check Routes

System health and status endpoints.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.schemas import HealthResponse
from src.services.queue import get_queue_service
from src.services.storage import get_storage_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


def get_version() -> str:
    """Read version from VERSION file."""
    version_file = Path(__file__).parent.parent.parent.parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "unknown"


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns overall system health and individual service statuses.
    """
    services = {}
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        services["database"] = "healthy"
    except Exception as e:
        logger.error("Database health check failed: %s", type(e).__name__)
        services["database"] = "unhealthy"
    
    # Check Redis
    try:
        queue = get_queue_service()
        if queue.health_check():
            services["redis"] = "healthy"
        else:
            services["redis"] = "unhealthy"
    except Exception as e:
        logger.error("Redis health check failed: %s", type(e).__name__)
        services["redis"] = "unhealthy"
    
    # Check MinIO
    try:
        storage = get_storage_service()
        # Try to list bucket (simple operation)
        storage.list_files(prefix="health-check/")
        services["storage"] = "healthy"
    except Exception as e:
        logger.error("Storage health check failed: %s", type(e).__name__)
        services["storage"] = "unhealthy"
    
    # Overall status
    all_healthy = all(s == "healthy" for s in services.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=get_version(),
        services=services,
    )


@router.get("/health/live")
async def liveness():
    """
    Kubernetes liveness probe.
    
    Returns 200 if the application is running.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """
    Kubernetes readiness probe.
    
    Returns 200 if the application is ready to serve requests.
    """
    try:
        # Quick database check
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        logger.error("Readiness check failed")
        return {"status": "not ready"}
