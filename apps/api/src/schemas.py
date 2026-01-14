"""
Pydantic Schemas

Request/Response models for the API.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Pack Schemas
# ============================================================================

class PackCreate(BaseModel):
    """Request to create a new pack."""
    style_presets: list[str] = Field(
        default=["corporate"],
        description="Style presets for generation",
        examples=[["corporate", "natural"]],
    )


class PackResponse(BaseModel):
    """Pack details response."""
    id: UUID
    status: str
    style_preset: str | None
    style_presets: list[str]
    trigger_token: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    photo_count: int = 0
    output_count: int = 0

    class Config:
        from_attributes = True


class PackListResponse(BaseModel):
    """List of packs response."""
    packs: list[PackResponse]
    total: int


# ============================================================================
# Photo Schemas
# ============================================================================

class PhotoResponse(BaseModel):
    """Photo details response."""
    id: UUID
    pack_id: UUID
    original_filename: str
    quality_score: float | None
    is_valid: str
    face_detected: str
    created_at: datetime
    # Note: s3_key not exposed, use presigned URL endpoint

    class Config:
        from_attributes = True


class PhotoUploadResponse(BaseModel):
    """Response after uploading photos."""
    uploaded: int
    photos: list[PhotoResponse]
    errors: list[str] = []


# ============================================================================
# Job Schemas
# ============================================================================

class JobResponse(BaseModel):
    """Job details response."""
    id: UUID
    pack_id: UUID
    job_type: str
    status: str
    progress: int
    current_step: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    class Config:
        from_attributes = True


class JobCreateResponse(BaseModel):
    """Response after creating a job."""
    job: JobResponse
    message: str


# ============================================================================
# Output Schemas
# ============================================================================

class OutputResponse(BaseModel):
    """Output image response."""
    id: UUID
    pack_id: UUID
    style_preset: str | None
    score: float | None
    is_selected: bool
    created_at: datetime
    # URLs are generated separately via presigned URLs

    class Config:
        from_attributes = True


class OutputListResponse(BaseModel):
    """List of outputs response."""
    outputs: list[OutputResponse]
    total: int
    selected_count: int


class OutputSelectRequest(BaseModel):
    """Request to select/deselect outputs."""
    output_ids: list[UUID]
    selected: bool = True


# ============================================================================
# Style Schemas
# ============================================================================

class StylePreset(BaseModel):
    """Style preset information."""
    id: str
    name: str
    description: str


class StyleListResponse(BaseModel):
    """List of available styles."""
    styles: list[StylePreset]


# ============================================================================
# Health/Status Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    services: dict[str, str] = {}


class GenerateRequest(BaseModel):
    """Request to start generation."""
    style_presets: list[str] | None = None  # Override pack styles
    num_images_per_style: int = Field(default=20, ge=5, le=50)
