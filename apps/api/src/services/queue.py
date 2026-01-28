"""
Queue Service

Redis-based job queue for background processing.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import redis

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QueueService:
    """
    Redis-based job queue service.
    
    Handles job queuing and status management for the GPU worker.
    
    Queue keys:
    - visage:jobs:pending - List of pending job IDs
    - visage:jobs:{id}:data - Job data hash
    - visage:jobs:{id}:status - Job status
    - visage:watermark:pending - Watermark removal jobs
    - visage:watermark:{id}:data - Watermark job data
    """

    QUEUE_PREFIX = "visage:jobs"
    PENDING_QUEUE = f"{QUEUE_PREFIX}:pending"
    PROCESSING_SET = f"{QUEUE_PREFIX}:processing"
    
    # Watermark removal queue
    WATERMARK_PREFIX = "visage:watermark"
    WATERMARK_QUEUE = f"{WATERMARK_PREFIX}:pending"

    def __init__(self):
        """Initialize Redis connection."""
        self.redis = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    def enqueue(
        self,
        job_id: UUID,
        job_type: str,
        pack_id: UUID,
        parameters: dict[str, Any] | None = None,
        priority: int = 0,
    ) -> bool:
        """
        Add a job to the queue.
        
        Args:
            job_id: Unique job identifier
            job_type: Type of job (validate, train, generate, upscale)
            pack_id: Associated pack ID
            parameters: Job-specific parameters
            priority: Job priority (higher = more urgent)
            
        Returns:
            True if job was queued successfully
        """
        try:
            job_key = f"{self.QUEUE_PREFIX}:{job_id}:data"
            
            job_data = {
                "id": str(job_id),
                "type": job_type,
                "pack_id": str(pack_id),
                "parameters": json.dumps(parameters or {}),
                "priority": priority,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Store job data
            self.redis.hset(job_key, mapping=job_data)
            
            # Add to pending queue (use sorted set for priority)
            # Score is negative priority so higher priority comes first
            self.redis.zadd(
                self.PENDING_QUEUE,
                {str(job_id): -priority},
            )
            
            logger.info(f"Enqueued job {job_id} type={job_type}")
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to enqueue job: {e}")
            return False

    def dequeue(self) -> dict[str, Any] | None:
        """
        Get the next job from the queue (for worker use).
        
        Returns:
            Job data dict or None if queue is empty
        """
        try:
            # Get highest priority job (lowest score)
            result = self.redis.zpopmin(self.PENDING_QUEUE, count=1)
            
            if not result:
                return None
            
            job_id = result[0][0]
            job_key = f"{self.QUEUE_PREFIX}:{job_id}:data"
            
            # Get job data
            job_data = self.redis.hgetall(job_key)
            
            if not job_data:
                logger.warning(f"Job {job_id} not found in data store")
                return None
            
            # Mark as processing
            self.redis.sadd(self.PROCESSING_SET, job_id)
            self.redis.hset(job_key, "status", "processing")
            
            # Parse parameters back to dict
            job_data["parameters"] = json.loads(job_data.get("parameters", "{}"))
            
            logger.info(f"Dequeued job {job_id}")
            return job_data
            
        except redis.RedisError as e:
            logger.error(f"Failed to dequeue job: {e}")
            return None

    def complete_job(
        self,
        job_id: UUID,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """
        Mark a job as completed or failed.
        
        Args:
            job_id: Job identifier
            result: Job result data (if successful)
            error: Error message (if failed)
            
        Returns:
            True if status was updated
        """
        try:
            job_key = f"{self.QUEUE_PREFIX}:{job_id}:data"
            
            updates = {
                "status": "failed" if error else "completed",
                "completed_at": datetime.utcnow().isoformat(),
            }
            
            if result:
                updates["result"] = json.dumps(result)
            if error:
                updates["error"] = error
            
            self.redis.hset(job_key, mapping=updates)
            self.redis.srem(self.PROCESSING_SET, str(job_id))
            
            status = "failed" if error else "completed"
            logger.info(f"Job {job_id} {status}")
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to complete job: {e}")
            return False

    def get_job_status(self, job_id: UUID) -> dict[str, Any] | None:
        """
        Get current job status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job data dict or None
        """
        try:
            job_key = f"{self.QUEUE_PREFIX}:{job_id}:data"
            job_data = self.redis.hgetall(job_key)
            
            if job_data:
                job_data["parameters"] = json.loads(job_data.get("parameters", "{}"))
                if "result" in job_data:
                    job_data["result"] = json.loads(job_data["result"])
            
            return job_data or None
            
        except redis.RedisError as e:
            logger.error(f"Failed to get job status: {e}")
            return None

    def update_progress(
        self,
        job_id: UUID,
        progress: int,
        current_step: str | None = None,
    ) -> bool:
        """
        Update job progress.
        
        Args:
            job_id: Job identifier
            progress: Progress percentage (0-100)
            current_step: Current step description
            
        Returns:
            True if updated successfully
        """
        try:
            job_key = f"{self.QUEUE_PREFIX}:{job_id}:data"
            
            updates = {"progress": str(progress)}
            if current_step:
                updates["current_step"] = current_step
            
            self.redis.hset(job_key, mapping=updates)
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to update progress: {e}")
            return False

    def get_queue_length(self) -> int:
        """Get number of pending jobs."""
        try:
            return self.redis.zcard(self.PENDING_QUEUE)
        except redis.RedisError:
            return 0

    def get_processing_count(self) -> int:
        """Get number of jobs currently processing."""
        try:
            return self.redis.scard(self.PROCESSING_SET)
        except redis.RedisError:
            return 0

    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            self.redis.ping()
            return True
        except redis.RedisError:
            return False

    # =========================================================================
    # Watermark Removal Queue Methods
    # =========================================================================

    def enqueue_watermark_job(
        self,
        job_id: str,
        input_keys: list[str],
        pack_id: str | None = None,
    ) -> bool:
        """
        Add a watermark removal job to the queue.
        
        Args:
            job_id: Unique job identifier
            input_keys: List of S3 keys for input images
            pack_id: Optional pack ID if integrated with pack upload
            
        Returns:
            True if job was queued successfully
        """
        try:
            job_key = f"{self.WATERMARK_PREFIX}:{job_id}:data"
            
            job_data = {
                "id": job_id,
                "input_keys": json.dumps(input_keys),
                "pack_id": pack_id or "",
                "status": "pending",
                "progress": "0",
                "output_keys": json.dumps([]),
                "errors": json.dumps([]),
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Store job data
            self.redis.hset(job_key, mapping=job_data)
            
            # Add to pending queue
            self.redis.rpush(self.WATERMARK_QUEUE, job_id)
            
            logger.info(f"Enqueued watermark job {job_id} with {len(input_keys)} files")
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to enqueue watermark job: {e}")
            return False

    def dequeue_watermark_job(self) -> dict[str, Any] | None:
        """
        Get the next watermark removal job from the queue.
        
        Returns:
            Job data dict or None if queue is empty
        """
        try:
            # Pop from queue (blocking with timeout)
            result = self.redis.lpop(self.WATERMARK_QUEUE)
            
            if not result:
                return None
            
            job_id = result
            job_key = f"{self.WATERMARK_PREFIX}:{job_id}:data"
            
            # Get job data
            job_data = self.redis.hgetall(job_key)
            
            if not job_data:
                logger.warning(f"Watermark job {job_id} not found in data store")
                return None
            
            # Mark as processing
            self.redis.hset(job_key, "status", "processing")
            
            # Parse JSON fields
            job_data["input_keys"] = json.loads(job_data.get("input_keys", "[]"))
            job_data["output_keys"] = json.loads(job_data.get("output_keys", "[]"))
            job_data["errors"] = json.loads(job_data.get("errors", "[]"))
            
            logger.info(f"Dequeued watermark job {job_id}")
            return job_data
            
        except redis.RedisError as e:
            logger.error(f"Failed to dequeue watermark job: {e}")
            return None

    def get_watermark_job_status(self, job_id: str) -> dict[str, Any] | None:
        """
        Get current watermark job status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job data dict or None
        """
        try:
            job_key = f"{self.WATERMARK_PREFIX}:{job_id}:data"
            job_data = self.redis.hgetall(job_key)
            
            if job_data:
                job_data["input_keys"] = json.loads(job_data.get("input_keys", "[]"))
                job_data["output_keys"] = json.loads(job_data.get("output_keys", "[]"))
                job_data["errors"] = json.loads(job_data.get("errors", "[]"))
                job_data["progress"] = int(job_data.get("progress", 0))
            
            return job_data or None
            
        except redis.RedisError as e:
            logger.error(f"Failed to get watermark job status: {e}")
            return None

    def update_watermark_job(
        self,
        job_id: str,
        status: str | None = None,
        progress: int | None = None,
        output_keys: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> bool:
        """
        Update watermark job status.
        
        Args:
            job_id: Job identifier
            status: New status (pending, processing, completed, failed)
            progress: Progress percentage (0-100)
            output_keys: List of output S3 keys
            errors: List of error messages
            
        Returns:
            True if updated successfully
        """
        try:
            job_key = f"{self.WATERMARK_PREFIX}:{job_id}:data"
            
            updates = {}
            if status is not None:
                updates["status"] = status
            if progress is not None:
                updates["progress"] = str(progress)
            if output_keys is not None:
                updates["output_keys"] = json.dumps(output_keys)
            if errors is not None:
                updates["errors"] = json.dumps(errors)
            if status in ("completed", "failed"):
                updates["completed_at"] = datetime.utcnow().isoformat()
            
            if updates:
                self.redis.hset(job_key, mapping=updates)
            
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to update watermark job: {e}")
            return False

    def delete_watermark_job(self, job_id: str) -> bool:
        """
        Delete a watermark job and its data.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            job_key = f"{self.WATERMARK_PREFIX}:{job_id}:data"
            self.redis.delete(job_key)
            self.redis.lrem(self.WATERMARK_QUEUE, 0, job_id)
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to delete watermark job: {e}")
            return False

    def get_watermark_queue_length(self) -> int:
        """Get number of pending watermark jobs."""
        try:
            return self.redis.llen(self.WATERMARK_QUEUE)
        except redis.RedisError:
            return 0


# Singleton instance
_queue_service: QueueService | None = None


def get_queue_service() -> QueueService:
    """Get or create queue service singleton."""
    global _queue_service
    if _queue_service is None:
        _queue_service = QueueService()
    return _queue_service
