"""
Job Model

Background jobs for training and generation.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from src.database import Base


class JobType(str, Enum):
    """Job type values."""
    VALIDATE = "validate"  # Photo validation
    TRAIN = "train"        # LoRA training
    GENERATE = "generate"  # Image generation
    UPSCALE = "upscale"    # Post-processing


class JobStatus(str, Enum):
    """Job status values."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """
    Background job model.
    
    Jobs are queued in Redis and processed by the GPU worker.
    Each job tracks:
    - Type (validate, train, generate, upscale)
    - Status and progress
    - Logs and errors
    - Timing information
    """
    
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pack_id = Column(UUID(as_uuid=True), ForeignKey("packs.id"), nullable=False, index=True)
    
    # Job configuration
    job_type = Column(String(50), nullable=False, index=True)
    status = Column(String(50), default=JobStatus.PENDING, nullable=False, index=True)
    priority = Column(Integer, default=0)  # Higher = more urgent
    
    # Progress tracking
    progress = Column(Integer, default=0)  # 0-100 percentage
    current_step = Column(String(255), nullable=True)
    
    # Logs and errors
    logs = Column(JSONB, default=list)  # List of log entries
    error_message = Column(Text, nullable=True)
    
    # Job-specific parameters
    parameters = Column(JSONB, default=dict)
    result = Column(JSONB, default=dict)  # Job output/result data
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Worker tracking
    worker_id = Column(String(100), nullable=True)
    
    # Relationships
    pack = relationship("Pack", back_populates="jobs")
    outputs = relationship("Output", back_populates="job")

    def __repr__(self):
        return f"<Job {self.id} type={self.job_type} status={self.status}>"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
