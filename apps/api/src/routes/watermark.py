"""
Watermark Removal Routes

Endpoints for removing watermarks from images.
Provides both standalone tool and integration with pack upload.
"""
from __future__ import annotations

import io
import logging
import re
import uuid
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import get_settings
from src.services.storage import get_storage_service
from src.services.queue import get_queue_service

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/watermark", tags=["watermark"])


def sanitize_for_log(value: str) -> str:
    """Sanitize a value for safe logging (prevent log injection)."""
    return re.sub(r'[\r\n\t]', '', str(value))


# ============================================================================
# Schemas
# ============================================================================

class WatermarkJobResponse(BaseModel):
    """Response after queuing watermark removal job."""
    job_id: str
    status: str
    message: str
    input_count: int
    input_keys: list[str]


class WatermarkStatusResponse(BaseModel):
    """Status of a watermark removal job."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    output_keys: list[str]
    output_urls: list[str]
    errors: list[str]


class WatermarkResultItem(BaseModel):
    """Single watermark removal result."""
    original_filename: str
    s3_key: str
    url: str
    watermark_detected: bool
    confidence: float


class WatermarkRemoveResponse(BaseModel):
    """Response after watermark removal (synchronous)."""
    processed: int
    results: list[WatermarkResultItem]
    errors: list[str]


# ============================================================================
# Standalone Watermark Removal
# ============================================================================

@router.post("/remove", response_model=WatermarkJobResponse)
async def remove_watermarks(
    files: Annotated[list[UploadFile], File(description="Images to process")],
) -> WatermarkJobResponse:
    """
    Remove watermarks from uploaded images.
    
    Uploads images to temporary storage and queues a watermark removal job.
    Poll the status endpoint to check progress and get results.
    
    Accepts: JPEG, PNG, WebP (max 20MB each, max 20 files)
    """
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files per request"
        )
    
    storage = get_storage_service()
    queue = get_queue_service()
    
    job_id = str(uuid.uuid4())
    input_keys = []
    errors = []
    
    for file in files:
        try:
            # Validate file extension
            ext = Path(file.filename or "").suffix.lower()
            if ext not in settings.allowed_extensions:
                errors.append(f"{file.filename}: Invalid file type (allowed: jpg, png, webp)")
                continue
            
            # Validate file size
            content = await file.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > settings.max_upload_size_mb:
                errors.append(f"{file.filename}: File too large ({size_mb:.1f}MB, max: {settings.max_upload_size_mb}MB)")
                continue
            
            # Generate S3 key for temporary storage
            file_id = uuid.uuid4()
            s3_key = f"watermark-jobs/{job_id}/input/{file_id}{ext}"
            
            # Upload to storage
            storage.upload_bytes(
                content,
                s3_key,
                content_type=file.content_type or "image/jpeg",
            )
            
            input_keys.append(s3_key)
            
        except Exception as e:
            logger.error("Failed to upload file for watermark removal: %s", type(e).__name__)
            errors.append(f"{file.filename}: Upload failed")
    
    if not input_keys:
        raise HTTPException(
            status_code=400,
            detail=f"No valid files uploaded. Errors: {errors}"
        )
    
    # Queue the watermark removal job
    queue.enqueue_watermark_job(
        job_id=job_id,
        input_keys=input_keys,
    )
    
    logger.info(
        "Queued watermark removal job %s with %d files",
        sanitize_for_log(job_id),
        len(input_keys)
    )
    
    return WatermarkJobResponse(
        job_id=job_id,
        status="pending",
        message=f"Watermark removal queued. Processing {len(input_keys)} images.",
        input_count=len(input_keys),
        input_keys=input_keys,
    )


@router.get("/status/{job_id}", response_model=WatermarkStatusResponse)
async def get_watermark_status(job_id: str) -> WatermarkStatusResponse:
    """
    Get status of a watermark removal job.
    
    Poll this endpoint to check progress and retrieve results when complete.
    """
    queue = get_queue_service()
    storage = get_storage_service()
    
    # Get job status from Redis
    status_data = queue.get_watermark_job_status(job_id)
    
    if status_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Generate URLs for completed outputs
    output_urls = []
    for key in status_data.get("output_keys", []):
        try:
            # Use public URL for direct browser access
            url = f"http://localhost:9000/{settings.minio_bucket}/{key}"
            output_urls.append(url)
        except Exception:
            pass
    
    return WatermarkStatusResponse(
        job_id=job_id,
        status=status_data.get("status", "unknown"),
        progress=status_data.get("progress", 0),
        output_keys=status_data.get("output_keys", []),
        output_urls=output_urls,
        errors=status_data.get("errors", []),
    )


@router.get("/download/{job_id}")
async def download_watermark_results(job_id: str):
    """
    Download all processed images from a completed job as a ZIP file.
    """
    queue = get_queue_service()
    storage = get_storage_service()
    
    # Get job status
    status_data = queue.get_watermark_job_status(job_id)
    
    if status_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if status_data.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Status: {status_data.get('status')}"
        )
    
    output_keys = status_data.get("output_keys", [])
    
    if not output_keys:
        raise HTTPException(status_code=404, detail="No output files found")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for key in output_keys:
            try:
                # Download image from MinIO
                image_data = storage.download_file(key)
                
                # Use original filename from key
                filename = Path(key).name
                zip_file.writestr(filename, image_data)
                
            except Exception as e:
                logger.warning(f"Failed to add {key} to ZIP: {e}")
    
    zip_buffer.seek(0)
    
    # Return as streaming response
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=watermark-removed-{job_id[:8]}.zip"
        }
    )


@router.delete("/job/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_watermark_job(job_id: str):
    """
    Delete a watermark removal job and its files.
    
    Cleans up temporary storage and removes job from queue.
    """
    queue = get_queue_service()
    storage = get_storage_service()
    
    # Delete files from storage
    prefix = f"watermark-jobs/{job_id}/"
    files = storage.list_files(prefix=prefix)
    for file in files:
        storage.delete_file(file["Key"])
    
    # Remove job from Redis
    queue.delete_watermark_job(job_id)
    
    logger.info("Deleted watermark job %s", sanitize_for_log(job_id))
