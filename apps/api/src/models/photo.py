"""
Photo Model

Uploaded user photos for LoRA training.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from src.database import Base


class Photo(Base):
    """
    Uploaded photo model.
    
    Photos are uploaded by users for LoRA training.
    Each photo is validated for:
    - Face detection
    - Image quality
    - Diversity (angles, expressions)
    """
    
    __tablename__ = "photos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pack_id = Column(UUID(as_uuid=True), ForeignKey("packs.id"), nullable=False, index=True)
    
    # File info
    s3_key = Column(Text, nullable=False)
    original_filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=True)
    file_size = Column(Float, nullable=True)  # In bytes
    
    # Validation results
    quality_score = Column(Float, nullable=True)  # 0.0 to 1.0
    is_valid = Column(String(20), default="pending")  # pending, valid, invalid
    validation_errors = Column(JSONB, default=list)
    
    # Face detection metadata
    face_detected = Column(String(20), default="pending")  # pending, yes, no
    face_bbox = Column(JSONB, nullable=True)  # Bounding box coordinates
    face_landmarks = Column(JSONB, nullable=True)
    
    # Image metadata
    image_metadata = Column(JSONB, default=dict)  # EXIF, dimensions, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    pack = relationship("Pack", back_populates="photos")

    def __repr__(self):
        return f"<Photo {self.id} pack={self.pack_id} valid={self.is_valid}>"
