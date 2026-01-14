"""
Post-Processing Pipeline

Enhance and polish generated headshots for final delivery.
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance
import cv2

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""
    # Face restoration
    enable_face_restoration: bool = True
    face_fidelity: float = 0.7  # 0-1, higher = more faithful to input
    
    # Upscaling
    enable_upscaling: bool = True
    upscale_factor: int = 2  # 1024 -> 2048
    
    # Color correction
    enable_color_correction: bool = True
    brightness_adjust: float = 1.0  # 1.0 = no change
    contrast_adjust: float = 1.05  # Slight boost
    saturation_adjust: float = 1.0


class PostProcessor:
    """
    Post-processing pipeline for generated headshots.
    
    Pipeline:
    1. Face restoration (CodeFormer/GFPGAN)
    2. Upscaling (Real-ESRGAN)
    3. Color correction
    """
    
    def __init__(self, config: Optional[PostProcessConfig] = None):
        """Initialize post-processor."""
        self.config = config or PostProcessConfig()
        self.face_restorer = None
        self.upscaler = None
        self._models_loaded = False
        
    def _load_models(self):
        """Load post-processing models lazily."""
        if self._models_loaded:
            return
            
        logger.info("Loading post-processing models...")
        
        # Try to load CodeFormer
        if self.config.enable_face_restoration:
            self._load_face_restorer()
            
        # Try to load Real-ESRGAN
        if self.config.enable_upscaling:
            self._load_upscaler()
            
        self._models_loaded = True
        
    def _load_face_restorer(self):
        """Load CodeFormer or GFPGAN face restoration model."""
        try:
            # Try CodeFormer first
            from codeformer import CodeFormer
            self.face_restorer = CodeFormer()
            self.face_restorer_type = "codeformer"
            logger.info("CodeFormer loaded for face restoration")
            return
        except ImportError:
            logger.info("CodeFormer not available, trying GFPGAN")
            
        try:
            # Fall back to GFPGAN
            from gfpgan import GFPGANer
            
            # GFPGAN requires model weights
            model_path = Path.home() / ".cache" / "gfpgan" / "GFPGANv1.4.pth"
            
            if model_path.exists():
                self.face_restorer = GFPGANer(
                    model_path=str(model_path),
                    upscale=1,  # Don't upscale, we do that separately
                    arch='clean',
                    channel_multiplier=2,
                )
                self.face_restorer_type = "gfpgan"
                logger.info("GFPGAN loaded for face restoration")
            else:
                logger.warning(f"GFPGAN weights not found at {model_path}")
                self.face_restorer = None
                
        except ImportError:
            logger.warning("Neither CodeFormer nor GFPGAN available")
            self.face_restorer = None
            
    def _load_upscaler(self):
        """Load Real-ESRGAN upscaler."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Model path
            model_path = Path.home() / ".cache" / "realesrgan" / "RealESRGAN_x2plus.pth"
            
            if model_path.exists():
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2
                )
                
                self.upscaler = RealESRGANer(
                    scale=2,
                    model_path=str(model_path),
                    model=model,
                    tile=0,  # No tiling for small images
                    tile_pad=10,
                    pre_pad=0,
                    half=False,  # Use float32 for MPS compatibility
                )
                logger.info("Real-ESRGAN loaded for upscaling")
            else:
                logger.warning(f"Real-ESRGAN weights not found at {model_path}")
                self.upscaler = None
                
        except ImportError:
            logger.warning("Real-ESRGAN not available, using Pillow upscaling")
            self.upscaler = None
            
    def restore_face(self, image: Image.Image) -> Image.Image:
        """
        Apply face restoration to enhance facial details.
        
        Args:
            image: Input image (PIL)
            
        Returns:
            Face-restored image
        """
        if not self.config.enable_face_restoration:
            return image
            
        self._load_models()
        
        if self.face_restorer is None:
            return image
            
        try:
            # Convert to numpy (BGR for OpenCV-based models)
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            if self.face_restorer_type == "codeformer":
                # CodeFormer restoration
                restored = self.face_restorer.restore(
                    img_array,
                    fidelity_weight=self.config.face_fidelity
                )
            elif self.face_restorer_type == "gfpgan":
                # GFPGAN restoration
                _, _, restored = self.face_restorer.enhance(
                    img_array,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
            else:
                return image
                
            # Convert back to PIL RGB
            restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
            return Image.fromarray(restored_rgb)
            
        except Exception as e:
            logger.warning(f"Face restoration failed: {e}")
            return image
            
    def upscale(self, image: Image.Image) -> Image.Image:
        """
        Upscale image using Real-ESRGAN or Pillow.
        
        Args:
            image: Input image (PIL)
            
        Returns:
            Upscaled image
        """
        if not self.config.enable_upscaling:
            return image
            
        if self.config.upscale_factor == 1:
            return image
            
        self._load_models()
        
        if self.upscaler is not None:
            try:
                # Convert to numpy BGR
                img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Upscale with Real-ESRGAN
                output, _ = self.upscaler.enhance(img_array, outscale=self.config.upscale_factor)
                
                # Convert back to PIL RGB
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                return Image.fromarray(output_rgb)
                
            except Exception as e:
                logger.warning(f"Real-ESRGAN upscaling failed: {e}, using Pillow")
                
        # Fallback to Pillow LANCZOS
        new_size = (
            image.width * self.config.upscale_factor,
            image.height * self.config.upscale_factor
        )
        return image.resize(new_size, Image.Resampling.LANCZOS)
        
    def color_correct(self, image: Image.Image) -> Image.Image:
        """
        Apply subtle color corrections.
        
        Args:
            image: Input image (PIL)
            
        Returns:
            Color-corrected image
        """
        if not self.config.enable_color_correction:
            return image
            
        result = image
        
        # Brightness adjustment
        if self.config.brightness_adjust != 1.0:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(self.config.brightness_adjust)
            
        # Contrast adjustment
        if self.config.contrast_adjust != 1.0:
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(self.config.contrast_adjust)
            
        # Saturation adjustment
        if self.config.saturation_adjust != 1.0:
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(self.config.saturation_adjust)
            
        return result
        
    def auto_color_correct(self, image: Image.Image) -> Image.Image:
        """
        Automatically adjust colors based on image analysis.
        
        Args:
            image: Input image (PIL)
            
        Returns:
            Color-corrected image
        """
        img_array = np.array(image)
        
        # Analyze current levels
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        # Auto-adjust brightness if needed
        if mean_brightness < 100:
            # Image is dark, brighten
            factor = min(1.3, 120 / mean_brightness)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        elif mean_brightness > 180:
            # Image is bright, darken slightly
            factor = max(0.85, 150 / mean_brightness)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
            
        # Check saturation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        mean_sat = np.mean(hsv[:, :, 1])
        
        if mean_sat < 80:
            # Undersaturated, boost slightly
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.15)
        elif mean_sat > 180:
            # Oversaturated, reduce
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.9)
            
        return image
        
    def sharpen(self, image: Image.Image, amount: float = 0.3) -> Image.Image:
        """
        Apply subtle sharpening.
        
        Args:
            image: Input image (PIL)
            amount: Sharpening amount (0.0-1.0)
            
        Returns:
            Sharpened image
        """
        if amount <= 0:
            return image
            
        enhancer = ImageEnhance.Sharpness(image)
        # Amount of 0.3 -> factor of 1.3
        return enhancer.enhance(1.0 + amount)
        
    def process(self, image: Image.Image) -> Image.Image:
        """
        Run full post-processing pipeline.
        
        Order:
        1. Face restoration
        2. Upscaling
        3. Color correction
        4. Light sharpening
        
        Args:
            image: Input image (PIL)
            
        Returns:
            Fully processed image
        """
        logger.info("Running post-processing pipeline...")
        
        # 1. Face restoration
        if self.config.enable_face_restoration:
            logger.debug("Applying face restoration")
            image = self.restore_face(image)
            
        # 2. Upscaling
        if self.config.enable_upscaling and self.config.upscale_factor > 1:
            logger.debug(f"Upscaling {self.config.upscale_factor}x")
            image = self.upscale(image)
            
        # 3. Color correction
        if self.config.enable_color_correction:
            logger.debug("Applying color correction")
            image = self.auto_color_correct(image)
            
        # 4. Light sharpening (after upscale)
        image = self.sharpen(image, amount=0.2)
        
        logger.info(f"Post-processing complete. Output size: {image.size}")
        return image
        
    def process_batch(self, images: list[dict]) -> list[dict]:
        """
        Process a batch of images.
        
        Args:
            images: List of dicts with 'image' key (PIL Image)
            
        Returns:
            Same list with processed images
        """
        for i, item in enumerate(images):
            image = item.get("image")
            if image is None:
                continue
                
            try:
                processed = self.process(image)
                item["image"] = processed
                item["processed"] = True
                logger.debug(f"Processed image {i+1}/{len(images)}")
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                item["processed"] = False
                
        return images
        
    def cleanup(self):
        """Release resources."""
        self.face_restorer = None
        self.upscaler = None
        self._models_loaded = False


def download_model_weights():
    """
    Download required model weights.
    
    Call this once during setup to download:
    - GFPGAN v1.4 weights
    - Real-ESRGAN x2plus weights
    """
    import urllib.request
    
    cache_dir = Path.home() / ".cache"
    
    # GFPGAN weights
    gfpgan_dir = cache_dir / "gfpgan"
    gfpgan_dir.mkdir(parents=True, exist_ok=True)
    gfpgan_path = gfpgan_dir / "GFPGANv1.4.pth"
    
    if not gfpgan_path.exists():
        logger.info("Downloading GFPGAN weights...")
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        urllib.request.urlretrieve(url, gfpgan_path)
        logger.info(f"Downloaded GFPGAN to {gfpgan_path}")
        
    # Real-ESRGAN weights
    esrgan_dir = cache_dir / "realesrgan"
    esrgan_dir.mkdir(parents=True, exist_ok=True)
    esrgan_path = esrgan_dir / "RealESRGAN_x2plus.pth"
    
    if not esrgan_path.exists():
        logger.info("Downloading Real-ESRGAN weights...")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        urllib.request.urlretrieve(url, esrgan_path)
        logger.info(f"Downloaded Real-ESRGAN to {esrgan_path}")
        
    logger.info("All model weights downloaded")
