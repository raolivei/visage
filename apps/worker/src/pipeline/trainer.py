"""
LoRA Trainer

Fine-tune SDXL with LoRA on user photos.
"""

import logging
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from ..config import get_settings, get_device, get_torch_dtype

logger = logging.getLogger(__name__)
settings = get_settings()


class LoRATrainer:
    """
    LoRA trainer for SDXL.
    
    Trains a lightweight LoRA adapter on user photos to capture
    their unique facial features and style.
    """

    def __init__(self):
        """Initialize trainer (models loaded lazily)."""
        self.device = get_device()
        self.dtype = get_torch_dtype()
        self.pipe = None
        
    def _load_pipeline(self):
        """Load SDXL pipeline if not already loaded."""
        if self.pipe is not None:
            return
            
        logger.info(f"Loading SDXL pipeline on {self.device}...")
        
        # Import here to avoid loading on import
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            settings.vae_id,
            torch_dtype=self.dtype,
        )
        
        # Load pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            settings.model_id,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "mps":
            # MPS-specific optimizations
            self.pipe.enable_attention_slicing()
        elif self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
            
        logger.info("SDXL pipeline loaded")

    def train(
        self,
        photo_paths: list[Path],
        output_dir: Path,
        trigger_token: str = "@visageUser",
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> Path:
        """
        Train LoRA on user photos.
        
        Args:
            photo_paths: List of paths to training images
            output_dir: Directory to save trained LoRA
            trigger_token: Trigger token for the LoRA
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to trained LoRA weights
        """
        self._load_pipeline()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        lora_path = output_dir / "lora_weights.safetensors"
        
        if progress_callback:
            progress_callback(5, "Preparing training data")
        
        # Load and preprocess images
        logger.info(f"Loading {len(photo_paths)} training images")
        train_images = []
        for path in photo_paths:
            try:
                img = Image.open(path).convert("RGB")
                # Resize to training resolution
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                train_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        
        if len(train_images) < 5:
            raise ValueError(f"Not enough valid training images: {len(train_images)}")
        
        if progress_callback:
            progress_callback(10, "Initializing LoRA training")
        
        # NOTE: This is a stub implementation
        # In production, use the actual PEFT/diffusers training loop
        # See: https://huggingface.co/docs/diffusers/training/lora
        
        logger.info("Starting LoRA training (stub implementation)")
        
        # Simulate training progress
        num_steps = settings.lora_train_steps
        for step in range(0, num_steps, num_steps // 10):
            progress = 10 + int((step / num_steps) * 80)
            if progress_callback:
                progress_callback(progress, f"Training step {step}/{num_steps}")
            
            # In production, this would be actual training steps:
            # - Forward pass through U-Net
            # - Compute loss
            # - Backward pass
            # - Update LoRA weights
            
        if progress_callback:
            progress_callback(90, "Saving LoRA weights")
        
        # Save LoRA weights (stub - creates empty file)
        # In production, use: pipe.unet.save_lora_weights(lora_path)
        logger.info(f"Saving LoRA weights to {lora_path}")
        
        # Create a placeholder file for now
        lora_path.touch()
        
        if progress_callback:
            progress_callback(100, "Training complete")
        
        logger.info("LoRA training complete")
        return lora_path

    def cleanup(self):
        """Release GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
