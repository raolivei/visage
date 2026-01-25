"""
SLBR Watermark Remover

Wrapper for the SLBR (Self-calibrated Localization and Background Refinement) model
for visible watermark removal. This is a neural network approach that works well
for complex watermarks like semi-transparent text overlays.

Reference:
- Liang et al. "Visible Watermark Removal via Self-calibrated Localization and 
  Background Refinement" ACM MM 2021
- GitHub: https://github.com/bcmi/SLBR-Visible-Watermark-Removal
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Path to SLBR model files
SLBR_DIR = Path(__file__).parent.parent.parent / "models" / "slbr"
SLBR_CHECKPOINT = SLBR_DIR / "checkpoints" / "slbr_clwd.pth.tar"


class SLBRArgs:
    """Minimal args object for SLBR model initialization.
    
    These values match the pretrained CLWD checkpoint.
    """
    def __init__(self):
        self.nets = 'slbr'
        self.input_size = 256
        self.crop_size = 256
        self.gan_norm = False
        self.mask_mode = 'cat'
        self.bg_mode = 'res_mask'
        self.use_refine = True
        self.k_refine = 3
        self.k_skip_stage = 3
        self.k_center = 2  # Must be 2 to match checkpoint
        self.project_mode = 'simple'
        self.sim_metric = 'cos'
        self.lr = 1e-3


class SLBRWatermarkRemover:
    """
    Neural network-based watermark remover using SLBR model.
    
    SLBR is specifically designed for visible watermark removal and handles
    complex watermarks like semi-transparent text overlays better than
    traditional image processing approaches.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize SLBR remover.
        
        Args:
            device: Device to run on ('cuda', 'mps', 'cpu'). Auto-detect if None.
        """
        self._model = None
        self._model_loaded = False
        
        if device is None:
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif torch.backends.mps.is_available():
                # SLBR may have issues with MPS, use CPU for stability
                self._device = 'cpu'
            else:
                self._device = 'cpu'
        else:
            self._device = device
            
        logger.info(f"SLBRWatermarkRemover will use device: {self._device}")
    
    def _load_model(self):
        """Load SLBR model lazily."""
        if self._model_loaded:
            return
            
        if not SLBR_DIR.exists():
            raise RuntimeError(
                f"SLBR model directory not found: {SLBR_DIR}\n"
                "Please clone the SLBR repository into models/slbr/"
            )
            
        if not SLBR_CHECKPOINT.exists():
            raise RuntimeError(
                f"SLBR checkpoint not found: {SLBR_CHECKPOINT}\n"
                "Please download the pretrained model from Google Drive"
            )
        
        logger.info("Loading SLBR watermark removal model...")
        
        # Save original sys.path and modules
        original_path = sys.path.copy()
        
        # Add SLBR to path at the front to take precedence
        slbr_path = str(SLBR_DIR)
        sys.path.insert(0, slbr_path)
        
        try:
            # Save visage's src modules before removing conflicting ones
            visage_src_modules = {k: v for k, v in sys.modules.items() 
                                  if k == 'src' or k.startswith('src.')}
            
            # Remove any existing 'src' modules that might conflict with SLBR
            for mod in list(visage_src_modules.keys()):
                del sys.modules[mod]
            
            # Import only the network architecture (not the model wrapper)
            # This avoids the old skimage dependencies in BasicModel
            from src.networks.resunet import SLBR
            
            # Create model with minimal args (while SLBR modules are loaded)
            args = SLBRArgs()
            self._model = SLBR(args=args, shared_depth=1, blocks=3, long_skip=True)
            
            # Load checkpoint
            checkpoint = torch.load(SLBR_CHECKPOINT, map_location=self._device, weights_only=False)
            
            # Handle potential DataParallel wrapping
            state_dict = checkpoint['state_dict']
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self._model.load_state_dict(new_state_dict, strict=False)
            self._model.to(self._device)
            self._model.eval()
            
            self._model_loaded = True
            logger.info(f"SLBR model loaded successfully on {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to load SLBR model: {e}")
            raise
        finally:
            # Restore original sys.path
            sys.path = original_path
            
            # Restore visage's src modules
            for mod_name, mod in visage_src_modules.items():
                if mod_name not in sys.modules:
                    sys.modules[mod_name] = mod
    
    def _preprocess(self, image: np.ndarray, target_size: int = 256) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Preprocess image for SLBR model.
        
        Args:
            image: BGR image as numpy array
            target_size: Target size for model input
            
        Returns:
            Tuple of (preprocessed tensor, original size)
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Convert BGR to RGB and normalize to [0, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        
        # Resize to target size
        tensor = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        return tensor, original_size
    
    def _postprocess(self, tensor: torch.Tensor, original_size: tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output back to image.
        
        Args:
            tensor: Model output tensor [1, C, H, W]
            original_size: Original image size (width, height)
            
        Returns:
            BGR image as numpy array
        """
        # Resize back to original size
        tensor = F.interpolate(tensor, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)
        
        # Convert to numpy [H, W, C]
        rgb = tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
        
        # Clip and convert to uint8
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        return bgr
    
    def remove_watermark(
        self, 
        image: np.ndarray,
        process_size: int = 512,
        num_passes: int = 1,
        remove_banner: bool = True,
        apply_smoothing: bool = True
    ) -> np.ndarray:
        """
        Remove watermark from image using SLBR model.
        
        Args:
            image: BGR image as numpy array
            process_size: Size to process at (larger = better quality but slower)
            num_passes: Number of SLBR passes (2 recommended for aggressive watermarks)
            remove_banner: Whether to detect and remove bottom banner text
            apply_smoothing: Whether to apply light smoothing to reduce artifacts
            
        Returns:
            Cleaned BGR image
        """
        self._load_model()
        
        result = image.copy()
        
        # Run SLBR for specified number of passes
        for pass_num in range(num_passes):
            # Preprocess
            input_tensor, original_size = self._preprocess(result, process_size)
            input_tensor = input_tensor.to(self._device).float()
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(input_tensor)
                
                # SLBR outputs: (background_predictions, mask_predictions, watermark_predictions)
                imoutput, immask_all, imwatermark = outputs
                
                # Get the refined output and mask
                imoutput = imoutput[0]  # First element is the refined output
                immask = immask_all[0]  # First element is the refined mask
                
                # Combine: output * mask + input * (1 - mask)
                imfinal = imoutput * immask + input_tensor * (1 - immask)
            
            # Postprocess
            result = self._postprocess(imfinal, original_size)
        
        # Remove bottom banner if requested
        if remove_banner:
            result = self._remove_bottom_banner(result)
        
        # Apply light smoothing if requested
        if apply_smoothing:
            result = cv2.bilateralFilter(result, d=5, sigmaColor=25, sigmaSpace=25)
        
        return result
    
    def _remove_bottom_banner(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and remove the bottom banner ("No watermark on download").
        
        Args:
            image: BGR image
            
        Returns:
            Image with banner removed via inpainting
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for banner in bottom 15%
        banner_region_start = int(h * 0.85)
        bottom_region = gray[banner_region_start:, :]
        
        # Find dark pixels (banner background)
        _, banner_binary = cv2.threshold(bottom_region, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(banner_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        banner_mask = np.zeros((h, w), dtype=np.uint8)
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            # Banner is wide and not too tall
            if cw > w * 0.15 and ch > 5 and ch < h * 0.1:
                y_full = banner_region_start + y
                pad = 5
                banner_mask[max(0, y_full-pad):min(h, y_full+ch+pad), 
                           max(0, x-pad):min(w, x+cw+pad)] = 255
        
        # Fallback: check for dark strip at bottom
        if np.sum(banner_mask > 0) < 100:
            very_bottom = gray[-int(h*0.05):, :]
            if np.mean(very_bottom) < 120:
                banner_mask[-int(h*0.06):, :] = 255
        
        # Apply inpainting if banner detected
        if np.sum(banner_mask > 0) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
            banner_mask = cv2.dilate(banner_mask, kernel, iterations=1)
            image = cv2.inpaint(image, banner_mask, 5, cv2.INPAINT_NS)
        
        return image
    
    def remove_watermark_enhanced(
        self, 
        image: np.ndarray,
        process_size: int = 1024
    ) -> np.ndarray:
        """
        Enhanced watermark removal with double pass at high resolution.
        
        Best quality but slower (~60-90 seconds on CPU).
        
        Args:
            image: BGR image
            process_size: Processing resolution (1024 recommended)
            
        Returns:
            Cleaned BGR image
        """
        return self.remove_watermark(
            image,
            process_size=process_size,
            num_passes=2,
            remove_banner=True,
            apply_smoothing=True
        )
    
    def _remove_diagonal_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Remove diagonal line artifacts using morphological operations.
        
        Targets the crosshatch scratch pattern common in watermarks.
        
        Args:
            image: BGR image
            
        Returns:
            Image with diagonal lines reduced
        """
        h, w = image.shape[:2]
        result = image.copy()
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create diagonal kernels for 45° and 135° lines
        kernel_size = 15
        
        # 45-degree line kernel
        kernel_45 = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        for i in range(kernel_size):
            kernel_45[i, i] = 1
        
        # 135-degree line kernel  
        kernel_135 = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        for i in range(kernel_size):
            kernel_135[i, kernel_size - 1 - i] = 1
        
        # Apply morphological top-hat to detect thin diagonal lines
        tophat_45 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_45)
        tophat_135 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_135)
        
        # Combine line detections
        lines_combined = cv2.add(tophat_45, tophat_135)
        
        # Threshold to get line mask
        _, line_mask = cv2.threshold(lines_combined, 15, 255, cv2.THRESH_BINARY)
        
        # Dilate slightly to cover the full line width
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        line_mask = cv2.dilate(line_mask, kernel_dilate, iterations=1)
        
        # Only process if significant lines detected (but not too much - avoid face)
        line_coverage = np.sum(line_mask > 0) / line_mask.size
        if 0.001 < line_coverage < 0.15:
            # Use median filter along detected lines
            # This blends the line pixels with surrounding pixels
            for c in range(3):
                channel = result[:, :, c].astype(np.float32)
                
                # Apply median filter
                median = cv2.medianBlur(result[:, :, c], 3)
                
                # Blend only on line regions
                mask_float = line_mask.astype(np.float32) / 255.0
                result[:, :, c] = (channel * (1 - mask_float) + median * mask_float).astype(np.uint8)
        
        return result
    
    def _apply_fft_cleanup(self, image: np.ndarray) -> np.ndarray:
        """
        Apply FFT-based cleanup to remove remaining periodic patterns.
        
        Analyzes frequency spectrum and applies notch filters to
        suppress any remaining periodic artifacts.
        
        Args:
            image: BGR image
            
        Returns:
            Image with periodic patterns reduced
        """
        from scipy.ndimage import gaussian_filter
        
        result = np.zeros_like(image)
        
        # Process each channel
        for c in range(3):
            channel = image[:, :, c].astype(np.float64)
            
            # Apply FFT
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            # Log transform for better visualization
            magnitude_log = np.log1p(magnitude)
            
            h, w = channel.shape
            center_y, center_x = h // 2, w // 2
            
            # Mask out center (DC and very low frequencies)
            center_mask_size = min(h, w) // 15
            magnitude_log[center_y-center_mask_size:center_y+center_mask_size, 
                          center_x-center_mask_size:center_x+center_mask_size] = 0
            
            # Find strong frequency peaks (potential periodic artifacts)
            threshold = np.percentile(magnitude_log, 99.0)
            peaks = magnitude_log > threshold
            
            # Count peaks - only filter if there are significant periodic components
            num_peaks = np.sum(peaks)
            
            if num_peaks > 10:
                # Create notch filter mask
                notch_mask = np.ones((h, w), dtype=np.float64)
                
                # Apply Gaussian blur to peak mask to create smooth notches
                peak_mask_float = peaks.astype(np.float64)
                notch_radius = 3
                smooth_notch = gaussian_filter(peak_mask_float, sigma=notch_radius)
                
                # Invert and apply as suppression
                notch_mask = 1.0 - np.clip(smooth_notch * 2, 0, 0.8)
                
                # Apply notch filter
                fshift_filtered = fshift * notch_mask
                
                # Inverse FFT
                f_ishift = np.fft.ifftshift(fshift_filtered)
                channel_filtered = np.fft.ifft2(f_ishift)
                result[:, :, c] = np.clip(np.abs(channel_filtered), 0, 255).astype(np.uint8)
            else:
                result[:, :, c] = channel.astype(np.uint8)
        
        return result
    
    def _detect_face_region(self, image: np.ndarray) -> tuple[int, int, int, int]:
        """
        Detect face region in headshot image.
        
        Uses a simple heuristic for headshots: face is typically
        in the center-upper portion of the image.
        
        Args:
            image: BGR image
            
        Returns:
            Tuple of (x, y, width, height) for face bounding box
        """
        h, w = image.shape[:2]
        
        # Try OpenCV face detection first
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                x, y, fw, fh = largest
                # Expand slightly
                pad = int(min(fw, fh) * 0.2)
                return (
                    max(0, x - pad),
                    max(0, y - pad),
                    min(w, fw + 2*pad),
                    min(h, fh + 2*pad)
                )
        except Exception:
            pass
        
        # Fallback: assume face is in center-upper region for headshots
        face_x = int(w * 0.2)
        face_y = int(h * 0.1)
        face_w = int(w * 0.6)
        face_h = int(h * 0.6)
        
        return (face_x, face_y, face_w, face_h)
    
    def _apply_face_aware_smoothing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply smoothing with face region protection.
        
        Applies stronger smoothing to background areas while
        preserving face details.
        
        Args:
            image: BGR image
            
        Returns:
            Smoothed image with face preserved
        """
        h, w = image.shape[:2]
        
        # Detect face region
        fx, fy, fw, fh = self._detect_face_region(image)
        
        # Create face mask (soft edges)
        face_mask = np.zeros((h, w), dtype=np.float32)
        face_mask[fy:fy+fh, fx:fx+fw] = 1.0
        
        # Blur the mask for soft transitions
        face_mask = cv2.GaussianBlur(face_mask, (51, 51), 0)
        
        # Apply stronger bilateral filter to the whole image
        strong_smooth = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
        
        # Apply gentle bilateral filter for face region
        gentle_smooth = cv2.bilateralFilter(image, d=5, sigmaColor=20, sigmaSpace=20)
        
        # Blend based on face mask
        result = np.zeros_like(image)
        for c in range(3):
            result[:, :, c] = (
                gentle_smooth[:, :, c] * face_mask + 
                strong_smooth[:, :, c] * (1 - face_mask)
            ).astype(np.uint8)
        
        return result
    
    def remove_watermark_aggressive(
        self, 
        image: np.ndarray,
        process_size: int = 1024,
        num_passes: int = 3
    ) -> np.ndarray:
        """
        Aggressive watermark removal combining multiple techniques.
        
        This is the highest quality option but slowest (~2-3 minutes on CPU).
        
        Pipeline:
        1. Triple SLBR passes at high resolution
        2. Directional line filter for diagonal artifacts
        3. FFT notch filter for periodic patterns
        4. Banner removal
        5. Face-aware smoothing
        
        Args:
            image: BGR image
            process_size: Processing resolution (1024 recommended)
            num_passes: Number of SLBR passes (3 recommended)
            
        Returns:
            Cleaned BGR image
        """
        logger.info(f"Starting aggressive watermark removal ({num_passes} SLBR passes)")
        
        # Step 1: Multiple SLBR passes
        logger.info(f"Step 1/{5}: Running {num_passes} SLBR passes at {process_size}px")
        result = self.remove_watermark(
            image,
            process_size=process_size,
            num_passes=num_passes,
            remove_banner=False,  # We'll do this later
            apply_smoothing=False  # We'll use face-aware smoothing instead
        )
        
        # Step 2: Remove diagonal line artifacts
        logger.info("Step 2/5: Removing diagonal line artifacts")
        result = self._remove_diagonal_lines(result)
        
        # Step 3: FFT cleanup for periodic patterns
        logger.info("Step 3/5: FFT cleanup for periodic patterns")
        result = self._apply_fft_cleanup(result)
        
        # Step 4: Remove bottom banner
        logger.info("Step 4/5: Removing bottom banner")
        result = self._remove_bottom_banner(result)
        
        # Step 5: Face-aware smoothing
        logger.info("Step 5/5: Applying face-aware smoothing")
        result = self._apply_face_aware_smoothing(result)
        
        logger.info("Aggressive watermark removal complete")
        return result
    
    def remove_watermark_pil(self, image: Image.Image, process_size: int = 512) -> Image.Image:
        """
        Remove watermark from PIL Image.
        
        Args:
            image: PIL Image to process
            process_size: Size to process at
            
        Returns:
            Cleaned PIL Image
        """
        # Convert to numpy BGR
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process
        result = self.remove_watermark(cv_image, process_size)
        
        # Convert back to PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def cleanup(self):
        """Release model resources."""
        self._model = None
        self._model_loaded = False
        logger.info("SLBRWatermarkRemover resources released")


def is_slbr_available() -> bool:
    """Check if SLBR model is available."""
    return SLBR_DIR.exists() and SLBR_CHECKPOINT.exists()
