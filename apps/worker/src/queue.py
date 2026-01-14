"""
Queue Client

Redis job queue client for worker.
"""

import json
import logging
from datetime import datetime
from typing import Any

import redis

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QueueClient:
    """Redis job queue client."""

    QUEUE_PREFIX = "visage:jobs"
    PENDING_QUEUE = f"{QUEUE_PREFIX}:pending"
    PROCESSING_SET = f"{QUEUE_PREFIX}:processing"

    def __init__(self):
        """Initialize Redis connection."""
        self.redis = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self.worker_id = settings.worker_id

    def dequeue(self) -> dict[str, Any] | None:
        """
        Get the next job from the queue.
        
        Returns:
            Job data dict or None if queue is empty
        """
        try:
            # Get highest priority job
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
            self.redis.hset(job_key, mapping={
                "status": "processing",
                "worker_id": self.worker_id,
                "started_at": datetime.utcnow().isoformat(),
            })
            
            # Parse parameters
            job_data["parameters"] = json.loads(job_data.get("parameters", "{}"))
            
            logger.info(f"Dequeued job {job_id} type={job_data.get('type')}")
            return job_data
            
        except redis.RedisError as e:
            logger.error(f"Failed to dequeue job: {e}")
            return None

    def complete_job(
        self,
        job_id: str,
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
            self.redis.srem(self.PROCESSING_SET, job_id)
            
            status = "failed" if error else "completed"
            logger.info(f"Job {job_id} {status}")
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to complete job: {e}")
            return False

    def update_progress(
        self,
        job_id: str,
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

    def add_log(self, job_id: str, message: str, level: str = "info") -> bool:
        """
        Add a log entry to a job.
        
        Args:
            job_id: Job identifier
            message: Log message
            level: Log level (info, warning, error)
            
        Returns:
            True if added successfully
        """
        try:
            logs_key = f"{self.QUEUE_PREFIX}:{job_id}:logs"
            
            log_entry = json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
            })
            
            self.redis.rpush(logs_key, log_entry)
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to add log: {e}")
            return False

    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            self.redis.ping()
            return True
        except redis.RedisError:
            return False


# Singleton
_queue_client: QueueClient | None = None


def get_queue_client() -> QueueClient:
    """Get or create queue client singleton."""
    global _queue_client
    if _queue_client is None:
        _queue_client = QueueClient()
    return _queue_client
