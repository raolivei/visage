"""
Visage AI Pipeline

Training, generation, and filtering modules.
"""

from .validator import PhotoValidator, ValidationResult, ProcessedPhoto, ValidationError
from .trainer import LoRATrainer
from .generator import ImageGenerator
from .filter import QualityFilter, QualityScores
from .postprocess import PostProcessor, PostProcessConfig
from .watermark_remover import WatermarkRemover, WatermarkDetector, WatermarkResult, DetectionResult

__all__ = [
    # Validation
    "PhotoValidator",
    "ValidationResult", 
    "ProcessedPhoto",
    "ValidationError",
    # Training
    "LoRATrainer", 
    # Generation
    "ImageGenerator", 
    # Quality
    "QualityFilter",
    "QualityScores",
    # Post-processing
    "PostProcessor",
    "PostProcessConfig",
    # Watermark removal
    "WatermarkRemover",
    "WatermarkDetector",
    "WatermarkResult",
    "DetectionResult",
]
