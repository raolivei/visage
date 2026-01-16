"""
Output Model

Generated headshot images.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from src.database import Base


class Output(Base):
    """
    Generated output image model.
    
    Outputs are generated images that passed quality filtering.
    Each output tracks:
    - S3 storage location
    - Prompt used for generation
    - Quality score
    - Selection status (user-selected for download)
    """
    
    __tablename__ = "outputs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pack_id = Column(UUID(as_uuid=True), ForeignKey("packs.id"), nullable=False, index=True)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=True, index=True)
    
    # File info
    s3_key = Column(Text, nullable=False)
    thumbnail_s3_key = Column(Text, nullable=True)  # Smaller preview
    
    # Generation metadata
    prompt_used = Column(Text, nullable=True)
    negative_prompt = Column(Text, nullable=True)
    style_preset = Column(Text, nullable=True)
    seed = Column(Float, nullable=True)  # For reproducibility
    
    # Quality metrics
    score = Column(Float, nullable=True)  # Overall quality score 0.0-1.0
    face_similarity = Column(Float, nullable=True)  # Similarity to reference photos
    artifact_score = Column(Float, nullable=True)  # Lower is better
    
    # Selection
    is_selected = Column(Boolean, default=False)  # User selected for download
    is_filtered_out = Column(Boolean, default=False)  # Failed quality check
    
    # Additional metadata
    generation_metadata = Column(JSONB, default=dict)  # Generation params, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    pack = relationship("Pack", back_populates="outputs")
    job = relationship("Job", back_populates="outputs")

    def __repr__(self):
        return f"<Output {self.id} pack={self.pack_id} score={self.score}>"
