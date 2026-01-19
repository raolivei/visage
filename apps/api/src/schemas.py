"""
Pydantic Schemas

Request/Response models for the API.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


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
    url: str  # Direct URL for browser access

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
    progress: int | None = 0
    current_step: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}
    
    @field_validator("progress", mode="before")
    @classmethod
    def progress_default(cls, v):
        return v if v is not None else 0


class JobCreateResponse(BaseModel):
    """Response after creating a job."""
    job: JobResponse
    message: str


class JobUpdateRequest(BaseModel):
    """Request to update job status (used by worker)."""
    status: str | None = None
    progress: int | None = None
    current_step: str | None = None
    error_message: str | None = None


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
    url: str  # Presigned URL for browser access

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


class OutputCreateItem(BaseModel):
    """Single output to create."""
    s3_key: str
    style_preset: str
    prompt_used: str | None = None
    negative_prompt: str | None = None
    seed: float | None = None
    score: float | None = None
    face_similarity: float | None = None
    artifact_score: float | None = None
    is_filtered_out: bool = False
    generation_metadata: dict | None = None


class OutputBatchCreateRequest(BaseModel):
    """Request to create multiple outputs at once."""
    job_id: UUID | None = None
    outputs: list[OutputCreateItem]


class OutputBatchCreateResponse(BaseModel):
    """Response after creating outputs."""
    created_count: int
    output_ids: list[UUID]


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
