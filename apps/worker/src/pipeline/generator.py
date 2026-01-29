"""
Image Generator

Generate headshots using SDXL with trained LoRA.
"""

import logging
from pathlib import Path
from typing import Callable
import random

import torch
from PIL import Image

from ..config import get_settings, get_device, get_torch_dtype

logger = logging.getLogger(__name__)
settings = get_settings()


class ImageGenerator:
    """
    SDXL image generator with LoRA support.
    
    Generates professional headshots using style-specific prompts
    and a user-trained LoRA adapter.
    """

    def __init__(self):
        """Initialize generator (models loaded lazily)."""
        self.device = get_device()
        self.dtype = get_torch_dtype()
        self.pipe = None
        self.current_lora: str | None = None

    def _load_pipeline(self):
        """Load SDXL pipeline if not already loaded."""
        if self.pipe is not None:
            return
            
        logger.info(f"Loading SDXL pipeline on {self.device}...")
        
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
            self.pipe.enable_attention_slicing()
        elif self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
            
        logger.info("SDXL pipeline loaded")

    def load_lora(self, lora_path: Path) -> bool:
        """
        Load LoRA weights into the pipeline.
        
        Handles PEFT-format LoRA files by converting key names to diffusers format.
        
        Args:
            lora_path: Path to LoRA weights file
            
        Returns:
            True if loaded successfully
        """
        self._load_pipeline()
        
        lora_path_str = str(lora_path)
        
        if self.current_lora == lora_path_str:
            logger.info("LoRA already loaded")
            return True
        
        try:
            logger.info(f"Loading LoRA from {lora_path}")
            
            # Load and convert PEFT format to diffusers format
            from safetensors.torch import load_file, save_file
            import tempfile
            import os
            
            state_dict = load_file(str(lora_path))
            
            # Check if this is PEFT format (keys start with 'base_model.model.')
            sample_key = next(iter(state_dict.keys()))
            if sample_key.startswith("base_model.model."):
                logger.info("Converting PEFT format to diffusers format...")
                
                # Convert PEFT keys to diffusers format
                # PEFT: base_model.model.down_blocks.0.attentions.0...
                # Diffusers: unet.down_blocks.0.attentions.0...
                converted_dict = {}
                for key, value in state_dict.items():
                    # Remove 'base_model.model.' prefix and add 'unet.' prefix
                    new_key = key.replace("base_model.model.", "unet.")
                    converted_dict[new_key] = value
                
                # Save converted weights to temp file
                with tempfile.NamedTemporaryFile(
                    suffix=".safetensors", delete=False
                ) as tmp:
                    save_file(converted_dict, tmp.name)
                    converted_path = Path(tmp.name)
                
                logger.info(f"Converted {len(converted_dict)} keys to diffusers format")
                
                # Load from converted file
                self.pipe.load_lora_weights(
                    str(converted_path.parent),
                    weight_name=converted_path.name,
                )
                
                # Clean up temp file
                os.unlink(converted_path)
            else:
                # Already in diffusers format
                self.pipe.load_lora_weights(
                    str(lora_path.parent),
                    weight_name=lora_path.name,
                )
            
            # Set LoRA scale (0.7-0.8 is usually a good balance)
            # Don't fuse - keep LoRA separate for better stability
            self.pipe.set_adapters(["default"], adapter_weights=[0.8])
            
            self.current_lora = lora_path_str
            logger.info(f"✅ LoRA loaded with scale 0.8 from {lora_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        num_images: int = 1,
        seed: int | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> list[tuple[Image.Image, int]]:
        """
        Generate images from a prompt.
        
        Args:
            prompt: Generation prompt
            negative_prompt: Negative prompt
            num_images: Number of images to generate
            seed: Random seed (None for random)
            progress_callback: Optional progress callback
            
        Returns:
            List of (image, seed) tuples
        """
        self._load_pipeline()
        
        results = []
        
        for i in range(num_images):
            if progress_callback:
                progress = int((i / num_images) * 100)
                progress_callback(progress, f"Generating image {i + 1}/{num_images}")
            
            # Set seed
            img_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=self.device).manual_seed(img_seed)
            
            try:
                # Generate image
                # NOTE: This is the actual generation call
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=settings.num_inference_steps,
                    guidance_scale=settings.guidance_scale,
                    width=settings.image_width,
                    height=settings.image_height,
                    generator=generator,
                )
                
                image = output.images[0]
                results.append((image, img_seed))
                
                logger.info(f"Generated image {i + 1}/{num_images} (seed={img_seed})")
                
            except Exception as e:
                logger.error(f"Failed to generate image: {e}")
                continue
        
        if progress_callback:
            progress_callback(100, "Generation complete")
        
        return results

    def generate_batch(
        self,
        prompts: list[tuple[str, str, str]],  # (style_id, prompt, negative_prompt)
        num_per_style: int = 20,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> list[dict]:
        """
        Generate multiple images for multiple styles.
        
        Args:
            prompts: List of (style_id, prompt, negative_prompt) tuples
            num_per_style: Number of images per style
            progress_callback: Optional progress callback
            
        Returns:
            List of result dicts with style_id, image, seed, prompt
        """
        self._load_pipeline()
        
        total = len(prompts) * num_per_style
        results = []
        count = 0
        
        for style_id, prompt, negative_prompt in prompts:
            logger.info(f"Generating {num_per_style} images for style: {style_id}")
            
            for i in range(num_per_style):
                count += 1
                if progress_callback:
                    progress = int((count / total) * 100)
                    progress_callback(progress, f"Generating {style_id} ({i + 1}/{num_per_style})")
                
                seed = random.randint(0, 2**32 - 1)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                try:
                    output = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=settings.num_inference_steps,
                        guidance_scale=settings.guidance_scale,
                        width=settings.image_width,
                        height=settings.image_height,
                        generator=generator,
                    )
                    
                    results.append({
                        "style_id": style_id,
                        "image": output.images[0],
                        "seed": seed,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to generate image: {e}")
                    continue
        
        if progress_callback:
            progress_callback(100, "Generation complete")
        
        return results

    def cleanup(self):
        """Release GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.current_lora = None
            
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
