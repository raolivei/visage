"""
API Routes

FastAPI routers for Visage endpoints.
"""

from src.routes.packs import router as packs_router
from src.routes.health import router as health_router
from src.routes.watermark import router as watermark_router

__all__ = ["packs_router", "health_router", "watermark_router"]
