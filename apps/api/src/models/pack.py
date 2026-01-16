"""
Pack Model

A pack represents a headshot generation session with uploaded photos and generated outputs.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from src.database import Base


class PackStatus(str, Enum):
    """Pack status values."""
    CREATED = "created"
    UPLOADING = "uploading"
    VALIDATING = "validating"
    TRAINING = "training"
    GENERATING = "generating"
    FILTERING = "filtering"
    COMPLETED = "completed"
    FAILED = "failed"


class Pack(Base):
    """
    Headshot pack model.
    
    A pack contains:
    - Uploaded user photos
    - Selected style preset(s)
    - Generated output images
    - Associated training/generation jobs
    """
    
    __tablename__ = "packs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(50), default=PackStatus.CREATED, nullable=False, index=True)
    style_preset = Column(String(50), nullable=True)  # Main style, can have multiple
    style_presets = Column(JSONB, default=list)  # All selected styles
    
    # LoRA training metadata
    trigger_token = Column(String(100), nullable=True)
    lora_path = Column(Text, nullable=True)  # S3 path to trained LoRA
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    photos = relationship("Photo", back_populates="pack", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="pack", cascade="all, delete-orphan")
    outputs = relationship("Output", back_populates="pack", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Pack {self.id} status={self.status}>"
