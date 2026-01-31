"""
Minimal pipeline init for watermark-only worker.
Only exports watermark-related modules to avoid GPU dependencies.
"""

from .watermark_remover import WatermarkRemover, WatermarkDetector, WatermarkResult, DetectionResult

__all__ = [
    "WatermarkRemover",
    "WatermarkDetector", 
    "WatermarkResult",
    "DetectionResult",
]
