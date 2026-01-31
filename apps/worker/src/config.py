"""
Worker Configuration

Environment-based configuration for the GPU worker.
"""

import torch
from functools import lru_cache
from pydantic_settings import BaseSettings


# =============================================================================
# Model Presets - Portrait-optimized alternatives to vanilla SDXL
# =============================================================================

MODEL_PRESETS = {
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "description": "Vanilla SDXL - good general purpose",
    },
    "juggernaut": {
        "model_id": "RunDiffusion/Juggernaut-XL-v9",
        "description": "Juggernaut XL - excellent faces, photorealistic",
    },
    "realvis": {
        "model_id": "SG161222/RealVisXL_V4.0",
        "description": "RealVisXL - ultra photorealistic",
    },
    "leosam": {
        "model_id": "LEOSAM/HelloWorld-XL",
        "description": "LEOSAM HelloWorld - portrait-optimized",
    },
}


class Settings(BaseSettings):
    """Worker settings loaded from environment variables."""

    # Worker identity
    worker_id: str = "visage-worker-1"
    
    # API connection
    api_url: str = "http://localhost:8004"
    
    # Redis
    redis_url: str = "redis://localhost:6383"
    
    # MinIO / S3
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "visage"
    minio_secure: bool = False
    
    # Device configuration
    device: str = "auto"  # auto, mps, cuda, cpu
    
    # Model configuration
    model_preset: str = "sdxl"  # sdxl, juggernaut, realvis, leosam
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    
    @property
    def model_id(self) -> str:
        """Get model ID from preset, with fallback to SDXL."""
        preset = MODEL_PRESETS.get(self.model_preset, MODEL_PRESETS["sdxl"])
        return preset["model_id"]
    
    # LoRA training defaults
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_lr: float = 1e-4
    lora_train_steps: int = 1000
    lora_batch_size: int = 1
    
    # Generation defaults
    num_inference_steps: int = 40  # Increased for better quality (30 was default)
    guidance_scale: float = 7.5
    image_width: int = 1024
    image_height: int = 1024
    
    # Quality thresholds
    min_face_similarity: float = 0.7
    min_quality_score: float = 0.6
    max_artifact_score: float = 0.3
    
    # Polling
    poll_interval: float = 5.0  # seconds
    
    # Metrics
    pushgateway_url: str = ""  # Empty = local HTTP server, set to pushgateway URL for remote
    metrics_push_interval: float = 15.0  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_device() -> str:
    """
    Determine the best available device.
    
    Priority: CUDA > MPS > CPU
    """
    settings = get_settings()
    
    if settings.device != "auto":
        return settings.device
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_torch_dtype() -> torch.dtype:
    """
    Get the appropriate dtype for the device.
    
    - CUDA: float16 (faster, less memory)
    - MPS: float16 (supported on Apple Silicon)
    - CPU: float32 (float16 not well supported)
    """
    device = get_device()
    
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32
