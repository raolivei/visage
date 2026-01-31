"""
Pack Routes

Endpoints for managing headshot packs.
"""
from __future__ import annotations

import io
import logging
import re
import uuid
import zipfile
from datetime import datetime


def sanitize_for_log(value: str) -> str:
    """Sanitize a value for safe logging (prevent log injection)."""
    return re.sub(r'[\r\n\t]', '', str(value))
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database import get_db
from src.models import Job, Output, Pack, Photo
from src.models.job import JobStatus, JobType
from src.models.pack import PackStatus
from src.schemas import (
    GenerateRequest,
    JobCreateResponse,
    JobResponse,
    JobUpdateRequest,
    OutputBatchCreateRequest,
    OutputBatchCreateResponse,
    OutputListResponse,
    OutputRatingRequest,
    OutputRatingResponse,
    OutputResponse,
    OutputSelectRequest,
    PackCreate,
    PackListResponse,
    PackResponse,
    PackUpdate,
    PhotoResponse,
    PhotoUploadResponse,
    StyleListResponse,
    StylePreset,
)
from src.services.queue import get_queue_service
from src.services.storage import get_storage_service

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/packs", tags=["packs"])


# ============================================================================
# Helper functions
# ============================================================================

def pack_to_response(pack: Pack, photo_count: int = 0, output_count: int = 0) -> PackResponse:
    """Convert Pack model to response schema."""
    return PackResponse(
        id=pack.id,
        status=pack.status,
        style_preset=pack.style_preset,
        style_presets=pack.style_presets or [],
        trigger_token=pack.trigger_token,
        error_message=pack.error_message,
        created_at=pack.created_at,
        updated_at=pack.updated_at,
        photo_count=photo_count,
        output_count=output_count,
    )


# ============================================================================
# Pack CRUD
# ============================================================================

@router.post("", response_model=PackResponse, status_code=status.HTTP_201_CREATED)
async def create_pack(
    data: PackCreate,
    db: AsyncSession = Depends(get_db),
) -> PackResponse:
    """
    Create a new headshot pack.
    
    A pack is a container for uploaded photos and generated outputs.
    """
    # Generate unique trigger token for this pack's LoRA
    trigger_token = f"@visage_{uuid.uuid4().hex[:8]}"
    
    pack = Pack(
        status=PackStatus.CREATED,
        style_preset=data.style_presets[0] if data.style_presets else "corporate",
        style_presets=data.style_presets,
        trigger_token=trigger_token,
    )
    
    db.add(pack)
    await db.flush()
    
    logger.info("Created pack %s", pack.id)
    return pack_to_response(pack)


@router.get("", response_model=PackListResponse)
async def list_packs(
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
    offset: int = 0,
) -> PackListResponse:
    """
    List all packs.
    
    Returns packs ordered by creation date (newest first).
    """
    # Get total count
    count_result = await db.execute(select(func.count(Pack.id)))
    total = count_result.scalar() or 0
    
    # Get packs with counts
    result = await db.execute(
        select(Pack)
        .order_by(Pack.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    packs = result.scalars().all()
    
    # Get photo and output counts
    pack_responses = []
    for pack in packs:
        photo_count_result = await db.execute(
            select(func.count(Photo.id)).where(Photo.pack_id == pack.id)
        )
        photo_count = photo_count_result.scalar() or 0
        
        output_count_result = await db.execute(
            select(func.count(Output.id)).where(
                Output.pack_id == pack.id,
                Output.is_filtered_out == False,  # noqa: E712
            )
        )
        output_count = output_count_result.scalar() or 0
        
        pack_responses.append(pack_to_response(pack, photo_count, output_count))
    
    return PackListResponse(packs=pack_responses, total=total)


@router.get("/{pack_id}", response_model=PackResponse)
async def get_pack(
    pack_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> PackResponse:
    """Get pack details by ID."""
    result = await db.execute(
        select(Pack).where(Pack.id == pack_id)
    )
    pack = result.scalar_one_or_none()
    
    if not pack:
        raise HTTPException(status_code=404, detail="Pack not found")
    
    # Get counts
    photo_count_result = await db.execute(
        select(func.count(Photo.id)).where(Photo.pack_id == pack_id)
    )
    photo_count = photo_count_result.scalar() or 0
    
    output_count_result = await db.execute(
        select(func.count(Output.id)).where(
            Output.pack_id == pack_id,
            Output.is_filtered_out == False,  # noqa: E712
        )
    )
    output_count = output_count_result.scalar() or 0
    
    return pack_to_response(pack, photo_count, output_count)


@router.patch("/{pack_id}", response_model=PackResponse)
async def update_pack(
    pack_id: uuid.UUID,
    data: PackUpdate,
    db: AsyncSession = Depends(get_db),
) -> PackResponse:
    """Update pack status and metadata (used by worker)."""
    result = await db.execute(
        select(Pack).where(Pack.id == pack_id)
    )
    pack = result.scalar_one_or_none()
    
    if not pack:
        raise HTTPException(status_code=404, detail="Pack not found")
    
    # Update fields if provided
    if data.status is not None:
        pack.status = data.status
    if data.error_message is not None:
        pack.error_message = data.error_message
    if data.style_preset is not None:
        pack.style_preset = data.style_preset
    if data.style_presets is not None:
        pack.style_presets = data.style_presets
    
    await db.flush()
    
    logger.info(
        "Updated pack %s status=%s", 
        sanitize_for_log(pack_id), 
        sanitize_for_log(pack.status)
    )
    
    return pack_to_response(pack)


@router.delete("/{pack_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pack(
    pack_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a pack and all associated data."""
    result = await db.execute(
        select(Pack).where(Pack.id == pack_id)
    )
    pack = result.scalar_one_or_none()
    
    if not pack:
        raise HTTPException(status_code=404, detail="Pack not found")
    
    # Delete from storage (photos and outputs)
    storage = get_storage_service()
    files = storage.list_files(prefix=f"packs/{pack_id}/")
    for file in files:
        storage.delete_file(file["Key"])
    
    # Delete from database (cascades to photos, jobs, outputs)
    await db.delete(pack)
    
    logger.info("Deleted pack %s", sanitize_for_log(pack_id))


# ============================================================================
# Photo upload
# ============================================================================

@router.post("/{pack_id}/photos", response_model=PhotoUploadResponse)
async def upload_photos(
    pack_id: uuid.UUID,
    files: Annotated[list[UploadFile], File(description="Photos to upload")],
    db: AsyncSession = Depends(get_db),
    remove_watermarks: bool = False,
) -> PhotoUploadResponse:
    """
    Upload photos to a pack.
    
    Photos will be validated for:
    - File type (JPEG, PNG, WebP)
    - File size (max 20MB)
    - Face detection (done asynchronously)
    
    If remove_watermarks=true, photos will be processed to remove
    watermarks before being used for training.
    """
    # Get pack
    result = await db.execute(select(Pack).where(Pack.id == pack_id))
    pack = result.scalar_one_or_none()
    
    if not pack:
        raise HTTPException(status_code=404, detail="Pack not found")
    
    # Check current photo count
    count_result = await db.execute(
        select(func.count(Photo.id)).where(Photo.pack_id == pack_id)
    )
    current_count = count_result.scalar() or 0
    
    if current_count + len(files) > settings.max_photos_per_pack:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_photos_per_pack} photos per pack"
        )
    
    storage = get_storage_service()
    uploaded_photos = []
    errors = []
    watermark_input_keys = []  # Keys to process for watermark removal
    watermark_job_id = None
    
    for file in files:
        try:
            # Validate file extension
            ext = Path(file.filename or "").suffix.lower()
            if ext not in settings.allowed_extensions:
                errors.append(f"{file.filename}: Invalid file type")
                continue
            
            # Validate file size
            content = await file.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > settings.max_upload_size_mb:
                errors.append(f"{file.filename}: File too large ({size_mb:.1f}MB)")
                continue
            
            # Generate S3 key
            photo_id = uuid.uuid4()
            s3_key = f"packs/{pack_id}/photos/{photo_id}{ext}"
            
            # If watermark removal requested, store original separately
            original_s3_key = None
            if remove_watermarks:
                original_s3_key = f"packs/{pack_id}/photos/originals/{photo_id}{ext}"
                storage.upload_bytes(
                    content,
                    original_s3_key,
                    content_type=file.content_type or "image/jpeg",
                )
                watermark_input_keys.append(original_s3_key)
            
            # Upload to main location (will be replaced after watermark removal)
            storage.upload_bytes(
                content,
                s3_key,
                content_type=file.content_type or "image/jpeg",
            )
            
            # Create photo record
            photo = Photo(
                id=photo_id,
                pack_id=pack_id,
                s3_key=s3_key,
                original_filename=file.filename or "unknown",
                content_type=file.content_type,
                file_size=len(content),
                watermark_removed=False,
                original_s3_key=original_s3_key,
            )
            db.add(photo)
            uploaded_photos.append(photo)
            
        except Exception as e:
            logger.error("Failed to upload file: %s", type(e).__name__)
            errors.append(f"{file.filename}: Upload failed")
    
    # Queue watermark removal if requested
    if remove_watermarks and watermark_input_keys:
        queue = get_queue_service()
        watermark_job_id = str(uuid.uuid4())
        
        queue.enqueue_watermark_job(
            job_id=watermark_job_id,
            input_keys=watermark_input_keys,
            pack_id=str(pack_id),
        )
        
        # Update photos with job ID
        for photo in uploaded_photos:
            if photo.original_s3_key:
                photo.watermark_job_id = watermark_job_id
        
        logger.info(
            "Queued watermark removal for pack %s, job %s with %d files",
            sanitize_for_log(str(pack_id)),
            sanitize_for_log(watermark_job_id),
            len(watermark_input_keys)
        )
    
    # Update pack status
    if uploaded_photos:
        pack.status = PackStatus.UPLOADING
        await db.flush()
    
    # Convert to response
    photo_responses = [
        PhotoResponse(
            id=p.id,
            pack_id=p.pack_id,
            original_filename=p.original_filename,
            quality_score=p.quality_score,
            is_valid=p.is_valid,
            face_detected=p.face_detected,
            watermark_removed=p.watermark_removed,
            created_at=p.created_at,
            url=f"http://localhost:9000/{settings.minio_bucket}/{p.s3_key}" if p.s3_key else "",
        )
        for p in uploaded_photos
    ]
    
    return PhotoUploadResponse(
        uploaded=len(uploaded_photos),
        photos=photo_responses,
        errors=errors,
        watermark_job_id=watermark_job_id,
    )


@router.get("/{pack_id}/photos", response_model=list[PhotoResponse])
async def list_photos(
    pack_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[PhotoResponse]:
    """List all photos in a pack."""
    result = await db.execute(
        select(Photo)
        .where(Photo.pack_id == pack_id)
        .order_by(Photo.created_at)
    )
    photos = result.scalars().all()
    
    # Generate direct URLs for each photo (bucket is public)
    photo_responses = []
    for p in photos:
        url = f"http://localhost:9000/{settings.minio_bucket}/{p.s3_key}" if p.s3_key else ""
        photo_responses.append(
            PhotoResponse(
                id=p.id,
                pack_id=p.pack_id,
                original_filename=p.original_filename,
                quality_score=p.quality_score,
                is_valid=p.is_valid,
                face_detected=p.face_detected,
                watermark_removed=p.watermark_removed,
                created_at=p.created_at,
                url=url,
            )
        )
    
    return photo_responses


# ============================================================================
# Generation
# ============================================================================

@router.post("/{pack_id}/generate", response_model=JobCreateResponse)
async def start_generation(
    pack_id: uuid.UUID,
    data: GenerateRequest | None = None,
    db: AsyncSession = Depends(get_db),
) -> JobCreateResponse:
    """
    Start headshot generation for a pack.
    
    This will:
    1. Validate uploaded photos
    2. Train a LoRA model
    3. Generate headshots for each style
    4. Filter and deliver best results
    """
    # Get pack
    result = await db.execute(select(Pack).where(Pack.id == pack_id))
    pack = result.scalar_one_or_none()
    
    if not pack:
        raise HTTPException(status_code=404, detail="Pack not found")
    
    # Check photo count
    count_result = await db.execute(
        select(func.count(Photo.id)).where(Photo.pack_id == pack_id)
    )
    photo_count = count_result.scalar() or 0
    
    if photo_count < settings.min_photos_per_pack:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {settings.min_photos_per_pack} photos"
        )
    
    # Create training job
    job = Job(
        pack_id=pack_id,
        job_type=JobType.TRAIN,
        status=JobStatus.PENDING,
        parameters={
            "style_presets": data.style_presets if data else pack.style_presets,
            "num_images_per_style": data.num_images_per_style if data else 20,
        },
    )
    db.add(job)
    await db.flush()
    
    # Enqueue job
    queue = get_queue_service()
    queue.enqueue(
        job_id=job.id,
        job_type=job.job_type,
        pack_id=pack_id,
        parameters=job.parameters,
    )
    
    # Update pack status
    pack.status = PackStatus.TRAINING
    
    logger.info("Started generation for pack %s, job %s", sanitize_for_log(pack_id), sanitize_for_log(job.id))
    
    job_response = JobResponse(
        id=job.id,
        pack_id=job.pack_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
    
    return JobCreateResponse(
        job=job_response,
        message="Generation started. This may take 15-30 minutes.",
    )


@router.get("/{pack_id}/jobs", response_model=list[JobResponse])
async def list_jobs(
    pack_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[JobResponse]:
    """List all jobs for a pack."""
    result = await db.execute(
        select(Job)
        .where(Job.pack_id == pack_id)
        .order_by(Job.created_at.desc())
    )
    jobs = result.scalars().all()
    
    return [
        JobResponse(
            id=j.id,
            pack_id=j.pack_id,
            job_type=j.job_type,
            status=j.status,
            progress=j.progress,
            current_step=j.current_step,
            error_message=j.error_message,
            created_at=j.created_at,
            started_at=j.started_at,
            completed_at=j.completed_at,
        )
        for j in jobs
    ]


@router.patch("/{pack_id}/jobs/{job_id}", response_model=JobResponse)
async def update_job_status(
    pack_id: uuid.UUID,
    job_id: uuid.UUID,
    data: JobUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """
    Update job status (used by worker to sync status to database).
    
    This enables the worker to keep PostgreSQL in sync with Redis status.
    """
    result = await db.execute(
        select(Job).where(
            Job.id == job_id,
            Job.pack_id == pack_id,
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Update fields if provided
    if data.status is not None:
        job.status = data.status
        if data.status == "processing" and job.started_at is None:
            job.started_at = datetime.utcnow()
        elif data.status in ("completed", "failed"):
            job.completed_at = datetime.utcnow()
    
    if data.progress is not None:
        job.progress = data.progress
    
    if data.current_step is not None:
        job.current_step = data.current_step
    
    if data.error_message is not None:
        job.error_message = data.error_message
    
    await db.flush()
    
    logger.info(
        f"Updated job {sanitize_for_log(str(job_id))}: "
        f"status={job.status}, progress={job.progress}"
    )
    
    return JobResponse(
        id=job.id,
        pack_id=job.pack_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


# ============================================================================
# Outputs
# ============================================================================

@router.get("/{pack_id}/outputs", response_model=OutputListResponse)
async def list_outputs(
    pack_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    include_filtered: bool = False,
) -> OutputListResponse:
    """
    List generated outputs for a pack.
    
    By default, only shows outputs that passed quality filtering.
    """
    query = select(Output).where(Output.pack_id == pack_id)
    
    if not include_filtered:
        query = query.where(Output.is_filtered_out == False)  # noqa: E712
    
    query = query.order_by(Output.score.desc().nulls_last())
    
    result = await db.execute(query)
    outputs = result.scalars().all()
    
    # Generate direct URLs (bucket is public)
    output_responses = []
    for o in outputs:
        url = f"http://localhost:9000/{settings.minio_bucket}/{o.s3_key}" if o.s3_key else ""
        output_responses.append(
            OutputResponse(
                id=o.id,
                pack_id=o.pack_id,
                style_preset=o.style_preset,
                score=o.score,
                is_selected=o.is_selected,
                user_rating=o.user_rating,
                rating_reason=o.rating_reason,
                rated_at=o.rated_at,
                created_at=o.created_at,
                url=url,
            )
        )
    
    selected_count = sum(1 for o in outputs if o.is_selected)
    liked_count = sum(1 for o in outputs if o.user_rating is True)
    disliked_count = sum(1 for o in outputs if o.user_rating is False)
    
    return OutputListResponse(
        outputs=output_responses,
        total=len(output_responses),
        selected_count=selected_count,
        liked_count=liked_count,
        disliked_count=disliked_count,
    )


@router.post("/{pack_id}/outputs/select")
async def select_outputs(
    pack_id: uuid.UUID,
    data: OutputSelectRequest,
    db: AsyncSession = Depends(get_db),
):
    """Select or deselect outputs for download."""
    result = await db.execute(
        select(Output).where(
            Output.pack_id == pack_id,
            Output.id.in_(data.output_ids),
        )
    )
    outputs = result.scalars().all()
    
    for output in outputs:
        output.is_selected = data.selected
    
    return {"updated": len(outputs)}


@router.post("/{pack_id}/outputs/batch", response_model=OutputBatchCreateResponse)
async def create_outputs_batch(
    pack_id: uuid.UUID,
    data: OutputBatchCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Create multiple outputs at once (used by worker after generation).
    
    This enables incremental saving - outputs can be registered as each
    style completes rather than waiting for all generation to finish.
    """
    pack_result = await db.execute(select(Pack).where(Pack.id == pack_id))
    pack = pack_result.scalar_one_or_none()
    if not pack:
        raise HTTPException(status_code=404, detail="Pack not found")
    
    created_outputs = []
    for item in data.outputs:
        output = Output(
            pack_id=pack_id,
            job_id=data.job_id,
            s3_key=item.s3_key,
            style_preset=item.style_preset,
            prompt_used=item.prompt_used,
            negative_prompt=item.negative_prompt,
            seed=item.seed,
            score=item.score,
            face_similarity=item.face_similarity,
            artifact_score=item.artifact_score,
            is_filtered_out=item.is_filtered_out,
            generation_metadata=item.generation_metadata or {},
        )
        db.add(output)
        created_outputs.append(output)
    
    await db.flush()
    
    logger.info(
        "Created %s outputs for pack %s",
        len(created_outputs),
        sanitize_for_log(str(pack_id)),
    )
    
    return OutputBatchCreateResponse(
        created_count=len(created_outputs),
        output_ids=[o.id for o in created_outputs],
    )


@router.post("/{pack_id}/outputs/{output_id}/rate", response_model=OutputRatingResponse)
async def rate_output(
    pack_id: uuid.UUID,
    output_id: uuid.UUID,
    data: OutputRatingRequest,
    db: AsyncSession = Depends(get_db),
) -> OutputRatingResponse:
    """
    Rate an output image (thumbs up or thumbs down).
    
    This feedback helps improve future generations by:
    - Learning which styles work best
    - Improving quality filtering
    - Understanding user preferences
    """
    result = await db.execute(
        select(Output).where(
            Output.pack_id == pack_id,
            Output.id == output_id,
        )
    )
    output = result.scalar_one_or_none()
    
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")
    
    output.user_rating = data.rating
    output.rating_reason = data.reason
    output.rated_at = datetime.utcnow()
    
    await db.flush()
    
    sanitized_reason = (
        (data.reason or "").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    )
    safe_output_id = str(output_id).replace("\r", "").replace("\n", "")
    logger.info(
        "Rated output %s %s%s",
        safe_output_id,
        "👍" if data.rating else "👎",
        f" - {sanitized_reason}" if sanitized_reason else "",
    )
    
    return OutputRatingResponse(
        id=output.id,
        user_rating=output.user_rating,
        rating_reason=output.rating_reason,
        rated_at=output.rated_at,
        message="Thanks for your feedback!" if data.rating else "Thanks for the feedback - we'll try to improve!",
    )


@router.get("/{pack_id}/outputs/{output_id}/url")
async def get_output_url(
    pack_id: uuid.UUID,
    output_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get presigned URL for an output image."""
    result = await db.execute(
        select(Output).where(
            Output.pack_id == pack_id,
            Output.id == output_id,
        )
    )
    output = result.scalar_one_or_none()
    
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")
    
    storage = get_storage_service()
    url = storage.get_presigned_url(output.s3_key, expires_in=3600)
    
    return {"url": url, "expires_in": 3600}


@router.get("/{pack_id}/outputs/download")
async def download_outputs(
    pack_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    selected_only: bool = True,
):
    """
    Download selected outputs as a ZIP file.
    
    Creates a ZIP archive containing all selected (or all) outputs for the pack.
    """
    from fastapi.responses import StreamingResponse
    
    # Get outputs
    query = select(Output).where(Output.pack_id == pack_id)
    if selected_only:
        query = query.where(Output.is_selected == True)
    
    result = await db.execute(query)
    outputs = result.scalars().all()
    
    if not outputs:
        raise HTTPException(status_code=404, detail="No outputs found to download")
    
    storage = get_storage_service()
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for output in outputs:
            try:
                # Download image from MinIO
                image_data = storage.download_file(output.s3_key)
                
                # Create filename: style_seed.png
                filename = Path(output.s3_key).name
                if output.style_preset:
                    filename = f"{output.style_preset}_{filename}"
                
                zip_file.writestr(filename, image_data)
            except Exception as e:
                logger.warning(f"Failed to add {output.s3_key} to ZIP: {e}")
    
    zip_buffer.seek(0)
    
    # Return as streaming response
    pack_name = str(pack_id)[:8]
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=visage-{pack_name}-headshots.zip"
        }
    )


# ============================================================================
# Styles
# ============================================================================

@router.get("/styles", response_model=StyleListResponse)
async def list_styles() -> StyleListResponse:
    """List available style presets."""
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "packages"))
    from shared.prompts import get_available_styles
    
    styles = get_available_styles()
    return StyleListResponse(
        styles=[StylePreset(**s) for s in styles]
    )
