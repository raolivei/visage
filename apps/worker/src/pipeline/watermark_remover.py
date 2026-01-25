"""
Watermark Remover

Detects and removes watermarks from images using multiple strategies:
- LaMa inpainting for localized watermarks (corners, edges, banners)
- FFT frequency filtering for repeating patterns (tiled text like INSTAHEADSHOTS)

Designed for cleaning AI-generated headshots from online tools.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.signal import firwin, welch

from ..config import get_device

logger = logging.getLogger(__name__)


class WatermarkType(Enum):
    """Classification of watermark types for selecting removal strategy."""
    NONE = "none"                    # No watermark detected
    LOCALIZED = "localized"          # Corner, edge, or small area watermarks
    REPEATING_PATTERN = "repeating"  # Tiled/repeating text or patterns


@dataclass
class PatternAnalysis:
    """Result of repeating pattern analysis."""
    has_repeating_pattern: bool
    frequencies: list[float]  # Detected pattern frequencies (cycles/pixel)
    pattern_strength: float   # 0-1 strength of the pattern
    recommended_type: WatermarkType


class SemiTransparentWatermarkRemover:
    """
    Removes semi-transparent watermarks by estimating and reversing alpha blending.
    
    Many watermark services (like INSTAHEADSHOTS) add semi-transparent text overlays.
    These are NOT periodic patterns but alpha-blended elements.
    
    The watermark blending formula is:
        output = (1 - alpha) * original + alpha * watermark_color
    
    To recover the original:
        original = (output - alpha * watermark_color) / (1 - alpha)
    
    This class estimates alpha and watermark color from the image.
    """
    
    def __init__(self):
        """Initialize remover."""
        pass
    
    def estimate_watermark_params(
        self, 
        image: np.ndarray,
        watermark_color: tuple[int, int, int] = None
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """
        Estimate watermark alpha mask and color.
        
        Args:
            image: BGR image
            watermark_color: Optional known watermark color (BGR). Auto-detect if None.
            
        Returns:
            Tuple of (alpha_mask, watermark_color_bgr)
        """
        h, w = image.shape[:2]
        
        # Convert to float
        img_float = image.astype(np.float64) / 255.0
        
        # Watermarks are typically white/light gray semi-transparent overlays
        # Detect by looking for areas brighter than local context
        if watermark_color is None:
            # Estimate watermark color from bright regions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find high-frequency bright regions (text-like)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.abs(laplacian)
            
            # Normalize
            edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)
            
            # Regions with high edges AND high brightness are likely watermark
            bright = gray.astype(np.float64) / 255.0
            watermark_likelihood = edges * bright
            
            # Get color from high-likelihood regions
            mask = watermark_likelihood > np.percentile(watermark_likelihood, 95)
            if np.sum(mask) > 100:
                watermark_color = tuple(int(c) for c in np.mean(image[mask], axis=0))
            else:
                watermark_color = (200, 200, 200)  # Default light gray
        
        logger.info(f"Estimated watermark color (BGR): {watermark_color}")
        
        # Estimate alpha based on deviation from local mean
        # Areas with watermark will be shifted toward watermark color
        wm_color_float = np.array(watermark_color) / 255.0
        
        # Compute local mean using large kernel (background without watermark)
        kernel_size = 51
        local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
        
        # Estimate alpha: how much the pixel is shifted toward watermark color
        # alpha = (pixel - local_mean) / (wm_color - local_mean)
        alpha_per_channel = np.zeros((h, w, 3), dtype=np.float64)
        
        for c in range(3):
            denom = wm_color_float[c] - local_mean[:, :, c]
            # Avoid division by zero
            denom = np.where(np.abs(denom) < 0.01, 0.01 * np.sign(denom + 1e-6), denom)
            alpha_c = (img_float[:, :, c] - local_mean[:, :, c]) / denom
            alpha_per_channel[:, :, c] = np.clip(alpha_c, 0, 1)
        
        # Combine channels - take the max alpha (most confident detection)
        alpha = np.max(alpha_per_channel, axis=2)
        
        # Only keep significant alpha values (actual watermark regions)
        alpha = np.where(alpha > 0.05, alpha, 0)
        
        # Smooth the alpha mask
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        return alpha, watermark_color
    
    def remove_watermark(
        self,
        image: np.ndarray,
        alpha: np.ndarray = None,
        watermark_color: tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Remove semi-transparent watermark by reversing alpha blending.
        
        Args:
            image: BGR image
            alpha: Optional pre-computed alpha mask. Auto-estimate if None.
            watermark_color: Optional watermark color (BGR). Auto-detect if None.
            
        Returns:
            Cleaned image
        """
        if alpha is None:
            alpha, watermark_color = self.estimate_watermark_params(image, watermark_color)
        
        h, w = image.shape[:2]
        img_float = image.astype(np.float64) / 255.0
        wm_color_float = np.array(watermark_color) / 255.0
        
        # Reverse the alpha blending: original = (output - alpha * wm) / (1 - alpha)
        result = np.zeros_like(img_float)
        
        for c in range(3):
            # Avoid division by zero
            denom = 1.0 - alpha
            denom = np.where(denom < 0.1, 0.1, denom)
            
            result[:, :, c] = (img_float[:, :, c] - alpha * wm_color_float[c]) / denom
        
        # Clip to valid range
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def remove_watermark_adaptive(
        self,
        image: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Adaptive watermark removal with strength control.
        
        Uses multiple passes with different parameters for better results.
        
        Args:
            image: BGR image
            strength: Removal strength (0.0 to 2.0, default 1.0)
            
        Returns:
            Cleaned image
        """
        # First pass: estimate and remove primary watermark
        alpha, wm_color = self.estimate_watermark_params(image)
        
        # Scale alpha by strength
        alpha = np.clip(alpha * strength, 0, 0.9)
        
        result = self.remove_watermark(image, alpha, wm_color)
        
        # Second pass: handle diagonal line artifacts
        result = self._remove_line_artifacts(result)
        
        return result
    
    def _remove_line_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Remove thin diagonal line artifacts often present in watermarks."""
        # Detect thin lines using morphological operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create line detection kernels at different angles
        kernels = []
        for angle in [45, 135]:
            kernel = np.zeros((7, 7), dtype=np.uint8)
            cv2.line(kernel, (0, 0) if angle == 45 else (6, 0), 
                     (6, 6) if angle == 45 else (0, 6), 1, 1)
            kernels.append(kernel)
        
        # Detect lines
        line_mask = np.zeros_like(gray)
        for kernel in kernels:
            # Morphological operation to detect lines
            detected = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            line_mask = cv2.bitwise_or(line_mask, detected)
        
        # Threshold to get line regions
        _, line_binary = cv2.threshold(line_mask, 20, 255, cv2.THRESH_BINARY)
        
        # Dilate slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        line_binary = cv2.dilate(line_binary, kernel, iterations=1)
        
        # Use inpainting on line regions only (small regions, fast)
        if np.sum(line_binary > 0) > 0 and np.sum(line_binary > 0) < image.size * 0.1:
            result = cv2.inpaint(image, line_binary, 3, cv2.INPAINT_TELEA)
        else:
            result = image
        
        return result


class FFTPatternRemover:
    """
    Removes repeating watermark patterns using FFT-based frequency filtering.
    
    Best for: Periodic scan lines, grid patterns, regular tiled watermarks.
    NOT ideal for: Alpha-blended text overlays (use SemiTransparentWatermarkRemover).
    
    Based on:
    - Cannon, M. et al. (1983) "Background pattern removal by power spectral filtering"
    - Welch, P. (1967) "The use of fast fourier transform for estimation of power spectra"
    """
    
    # Minimum frequency to consider (cycles/pixel) - avoids DC component
    MIN_FREQUENCY = 1 / 50  # Patterns with period > 50 pixels ignored
    
    # Default filter parameters
    DEFAULT_NUM_TAPS = 65  # FIR filter length
    DEFAULT_EPS = 0.025    # Filter bandwidth parameter
    
    def __init__(self):
        """Initialize FFT pattern remover."""
        self._alpha_remover = SemiTransparentWatermarkRemover()
    
    def estimate_pattern_frequency(
        self, 
        image: np.ndarray,
        axis: int = 0,
        min_frequency: float = None
    ) -> tuple[float, float]:
        """
        Estimate the dominant repeating pattern frequency using Welch's method.
        
        Args:
            image: Grayscale or color image as numpy array
            axis: Axis to analyze (0=vertical patterns, 1=horizontal)
            min_frequency: Minimum frequency to consider (cycles/pixel)
            
        Returns:
            Tuple of (frequency, power) where frequency is in cycles/pixel
        """
        if min_frequency is None:
            min_frequency = self.MIN_FREQUENCY
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Sum along the perpendicular axis to get 1D signal
        if axis == 0:
            signal = gray.sum(axis=1).astype(np.float64)
        else:
            signal = gray.sum(axis=0).astype(np.float64)
            
        # Normalize
        signal = signal - signal.mean()
        
        # Use Welch's method to estimate power spectral density
        # fs=1 means frequency is in cycles/pixel
        frequencies, psd = welch(signal, fs=1.0, nperseg=min(256, len(signal)//4))
        
        # Zero out frequencies below minimum (DC and very low freq)
        psd[frequencies < min_frequency] = 0.0
        
        # Find peak frequency
        peak_idx = psd.argmax()
        peak_freq = frequencies[peak_idx]
        peak_power = psd[peak_idx]
        
        # Normalize power relative to total
        total_power = psd.sum()
        relative_power = peak_power / total_power if total_power > 0 else 0
        
        return peak_freq, relative_power
    
    def analyze_pattern(self, image: np.ndarray) -> PatternAnalysis:
        """
        Analyze image for repeating patterns and determine watermark type.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            PatternAnalysis with detected frequencies and recommendation
        """
        # Analyze both vertical and horizontal patterns
        v_freq, v_power = self.estimate_pattern_frequency(image, axis=0)
        h_freq, h_power = self.estimate_pattern_frequency(image, axis=1)
        
        # Also check diagonal patterns using FFT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        diag_strength = self._analyze_diagonal_frequency(gray)
        
        frequencies = []
        if v_power > 0.1:
            frequencies.append(v_freq)
        if h_power > 0.1:
            frequencies.append(h_freq)
            
        # Determine pattern strength
        pattern_strength = max(v_power, h_power, diag_strength)
        
        # Classify watermark type
        # Strong repeating pattern (>15% of spectral power) = use FFT
        has_repeating = pattern_strength > 0.15 and len(frequencies) > 0
        
        if not has_repeating and pattern_strength < 0.05:
            recommended_type = WatermarkType.NONE
        elif has_repeating:
            recommended_type = WatermarkType.REPEATING_PATTERN
        else:
            recommended_type = WatermarkType.LOCALIZED
            
        return PatternAnalysis(
            has_repeating_pattern=has_repeating,
            frequencies=frequencies,
            pattern_strength=pattern_strength,
            recommended_type=recommended_type
        )
    
    def _analyze_diagonal_frequency(self, gray: np.ndarray) -> float:
        """
        Analyze diagonal pattern strength using 2D FFT.
        
        Returns strength score (0-1).
        """
        h, w = gray.shape
        
        # Apply 2D FFT
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Log transform
        magnitude_log = np.log1p(magnitude)
        
        # Mask out center (DC and very low frequencies)
        center_y, center_x = h // 2, w // 2
        mask_size = min(h, w) // 20
        magnitude_log[center_y-mask_size:center_y+mask_size, 
                      center_x-mask_size:center_x+mask_size] = 0
        
        # Check for peaks along diagonals
        # Diagonal patterns appear as peaks along 45-degree lines in FFT
        diag1_sum = 0
        diag2_sum = 0
        total = 0
        
        for i in range(-min(h, w)//2 + mask_size, min(h, w)//2 - mask_size):
            y1 = center_y + i
            x1 = center_x + i
            y2 = center_y + i
            x2 = center_x - i
            
            if 0 <= y1 < h and 0 <= x1 < w:
                diag1_sum += magnitude_log[y1, x1]
            if 0 <= y2 < h and 0 <= x2 < w:
                diag2_sum += magnitude_log[y2, x2]
            total += 1
        
        if total == 0:
            return 0.0
            
        # Compare diagonal energy to overall energy
        total_energy = magnitude_log.sum()
        diag_energy = diag1_sum + diag2_sum
        
        return diag_energy / total_energy if total_energy > 0 else 0.0
    
    def remove_horizontal_pattern(
        self,
        image: np.ndarray,
        distortion_freq: float = None,
        num_taps: int = None,
        eps: float = None
    ) -> np.ndarray:
        """
        Remove horizontal repeating pattern using directional filtering.
        
        This applies a highpass filter vertically and lowpass horizontally
        to isolate and subtract the pattern.
        
        Args:
            image: Image as numpy array (grayscale or color)
            distortion_freq: Pattern frequency in cycles/pixel (auto-detect if None)
            num_taps: FIR filter length
            eps: Filter bandwidth parameter
            
        Returns:
            Filtered image with pattern removed
        """
        if num_taps is None:
            num_taps = self.DEFAULT_NUM_TAPS
        if eps is None:
            eps = self.DEFAULT_EPS
            
        image = np.asarray(image, dtype=np.float64)
        
        # Auto-detect frequency if not provided
        if distortion_freq is None:
            distortion_freq, _ = self.estimate_pattern_frequency(image, axis=0)
            logger.info(f"Auto-detected horizontal pattern frequency: {distortion_freq:.4f} cycles/pixel")
        
        if distortion_freq < self.MIN_FREQUENCY:
            logger.warning(f"Pattern frequency too low ({distortion_freq}), skipping filter")
            return image.astype(np.uint8) if image.max() > 1 else image
        
        # Adjust eps to ensure valid filter cutoff (must be > 0 and < fs/2)
        # The cutoff for highpass is (distortion_freq - eps), must be > 0
        # The cutoff for lowpass is eps, must be < 0.5
        adjusted_eps = min(eps, distortion_freq * 0.8, 0.49)  # Ensure cutoff stays valid
        hpf_cutoff = max(distortion_freq - adjusted_eps, 0.01)  # Minimum cutoff of 0.01
        lpf_cutoff = min(adjusted_eps, 0.49)  # Maximum cutoff of 0.49
        
        logger.debug(f"Filter cutoffs: HPF={hpf_cutoff:.4f}, LPF={lpf_cutoff:.4f}")
        
        # Design filters
        # Highpass to capture the pattern, lowpass to ensure it's pattern-like (not edges)
        hpf = firwin(num_taps, hpf_cutoff, pass_zero='highpass', fs=1)
        lpf = firwin(num_taps, lpf_cutoff, pass_zero='lowpass', fs=1)
        
        # Apply filters: subtract the isolated pattern
        # HPF vertical (axis 0), LPF horizontal (axis 1)
        if len(image.shape) == 3:
            # Process each channel
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                pattern = convolve1d(convolve1d(image[:,:,c], hpf, axis=0), lpf, axis=1)
                result[:,:,c] = image[:,:,c] - pattern
        else:
            pattern = convolve1d(convolve1d(image, hpf, axis=0), lpf, axis=1)
            result = image - pattern
            
        # Clip to valid range
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    def remove_vertical_pattern(
        self,
        image: np.ndarray,
        distortion_freq: float = None,
        num_taps: int = None,
        eps: float = None
    ) -> np.ndarray:
        """
        Remove vertical repeating pattern using directional filtering.
        
        Args:
            image: Image as numpy array
            distortion_freq: Pattern frequency (auto-detect if None)
            num_taps: FIR filter length
            eps: Filter bandwidth parameter
            
        Returns:
            Filtered image
        """
        if num_taps is None:
            num_taps = self.DEFAULT_NUM_TAPS
        if eps is None:
            eps = self.DEFAULT_EPS
            
        image = np.asarray(image, dtype=np.float64)
        
        if distortion_freq is None:
            distortion_freq, _ = self.estimate_pattern_frequency(image, axis=1)
            logger.info(f"Auto-detected vertical pattern frequency: {distortion_freq:.4f} cycles/pixel")
        
        if distortion_freq < self.MIN_FREQUENCY:
            logger.warning(f"Pattern frequency too low ({distortion_freq}), skipping filter")
            return image.astype(np.uint8) if image.max() > 1 else image
        
        # Adjust eps to ensure valid filter cutoff
        adjusted_eps = min(eps, distortion_freq * 0.8, 0.49)
        hpf_cutoff = max(distortion_freq - adjusted_eps, 0.01)
        lpf_cutoff = min(adjusted_eps, 0.49)
        
        logger.debug(f"Filter cutoffs: HPF={hpf_cutoff:.4f}, LPF={lpf_cutoff:.4f}")
        
        # Design filters - swap axes compared to horizontal
        hpf = firwin(num_taps, hpf_cutoff, pass_zero='highpass', fs=1)
        lpf = firwin(num_taps, lpf_cutoff, pass_zero='lowpass', fs=1)
        
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                pattern = convolve1d(convolve1d(image[:,:,c], hpf, axis=1), lpf, axis=0)
                result[:,:,c] = image[:,:,c] - pattern
        else:
            pattern = convolve1d(convolve1d(image, hpf, axis=1), lpf, axis=0)
            result = image - pattern
            
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
    
    def remove_diagonal_pattern(
        self,
        image: np.ndarray,
        num_taps: int = None,
        eps: float = None
    ) -> np.ndarray:
        """
        Remove diagonal repeating pattern using FFT notch filtering.
        
        This method uses 2D FFT to identify and suppress diagonal frequency components.
        
        Args:
            image: Image as numpy array
            num_taps: Not used directly but kept for API consistency
            eps: Bandwidth of notch filter (fraction of frequency)
            
        Returns:
            Filtered image
        """
        if eps is None:
            eps = self.DEFAULT_EPS * 2  # Wider for FFT approach
            
        image = np.asarray(image, dtype=np.float64)
        
        if len(image.shape) == 3:
            # Process each channel
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:,:,c] = self._fft_diagonal_filter(image[:,:,c], eps)
        else:
            result = self._fft_diagonal_filter(image, eps)
            
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
    
    def _fft_diagonal_filter(self, channel: np.ndarray, eps: float) -> np.ndarray:
        """Apply FFT-based diagonal notch filter to single channel."""
        h, w = channel.shape
        
        # Apply 2D FFT
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        
        # Create notch filter mask
        mask = np.ones((h, w), dtype=np.float64)
        center_y, center_x = h // 2, w // 2
        
        # Find and suppress diagonal peaks
        magnitude = np.abs(fshift)
        magnitude_log = np.log1p(magnitude)
        
        # Mask center
        center_size = min(h, w) // 20
        magnitude_log[center_y-center_size:center_y+center_size,
                      center_x-center_size:center_x+center_size] = 0
        
        # Find peaks (top 0.5% of frequencies)
        threshold = np.percentile(magnitude_log, 99.5)
        peak_mask = magnitude_log > threshold
        
        # Create smooth notch around peaks using Gaussian
        notch_radius = int(min(h, w) * eps)
        if notch_radius < 3:
            notch_radius = 3
            
        # Apply Gaussian blur to peak mask to create smooth notches
        peak_mask_float = peak_mask.astype(np.float64)
        from scipy.ndimage import gaussian_filter
        smooth_notch = gaussian_filter(peak_mask_float, sigma=notch_radius)
        
        # Invert to create suppression mask
        mask = 1.0 - np.clip(smooth_notch * 3, 0, 1)
        
        # Apply mask
        fshift_filtered = fshift * mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift_filtered)
        result = np.fft.ifft2(f_ishift)
        
        return np.abs(result)
    
    def remove_pattern(
        self,
        image: np.ndarray,
        analysis: PatternAnalysis = None,
        use_alpha_removal: bool = False
    ) -> np.ndarray:
        """
        Remove repeating pattern from image using FFT filtering.
        
        NOTE: This works best for simple periodic patterns (scan lines, grids).
        For complex semi-transparent text watermarks (like INSTAHEADSHOTS), 
        results may be limited. Consider using a trained neural network model
        for better results on complex watermarks.
        
        Args:
            image: BGR image as numpy array
            analysis: Optional pre-computed pattern analysis
            use_alpha_removal: If True, also try alpha-based removal (experimental)
            
        Returns:
            Filtered image with pattern reduced
        """
        if analysis is None:
            analysis = self.analyze_pattern(image)
            
        logger.info(f"Removing watermark pattern (strength: {analysis.pattern_strength:.2f})")
        
        result = image.copy()
        
        # Optionally try alpha-based removal first (experimental, may cause artifacts)
        if use_alpha_removal:
            logger.info("Step 1: Trying semi-transparent overlay removal (experimental)")
            result = self._alpha_remover.remove_watermark_adaptive(result, strength=0.5)
        
        # Apply FFT-based filtering
        if analysis.has_repeating_pattern:
            logger.info("Applying FFT filtering for periodic patterns")
            
            # Apply horizontal pattern removal
            v_freq, v_power = self.estimate_pattern_frequency(result, axis=0)
            if v_power > 0.1:
                logger.info(f"Removing horizontal pattern at {v_freq:.4f} cycles/pixel")
                result = self.remove_horizontal_pattern(result, v_freq)
            
            # Apply vertical pattern removal
            h_freq, h_power = self.estimate_pattern_frequency(result, axis=1)
            if h_power > 0.1:
                logger.info(f"Removing vertical pattern at {h_freq:.4f} cycles/pixel")
                result = self.remove_vertical_pattern(result, h_freq)
            
            # Apply diagonal pattern removal if strong diagonal component
            diag_strength = self._analyze_diagonal_frequency(
                cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
            )
            if diag_strength > 0.15:
                logger.info(f"Removing diagonal pattern (strength: {diag_strength:.2f})")
                result = self.remove_diagonal_pattern(result)
        
        return result
    
    def remove_pattern_from_pil(self, image: Image.Image) -> Image.Image:
        """
        Remove repeating pattern from PIL Image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Filtered PIL Image
        """
        # Convert to numpy
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process
        result = self.remove_pattern(cv_image)
        
        # Convert back to PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)


@dataclass
class WatermarkResult:
    """Result of watermark removal processing."""
    original_path: Path
    cleaned_image: Image.Image
    watermark_detected: bool
    mask_used: Optional[np.ndarray] = None
    confidence: float = 0.0
    watermark_type: WatermarkType = WatermarkType.NONE
    method_used: str = ""  # "inpaint" or "fft"


@dataclass
class DetectionResult:
    """Result of watermark detection."""
    detected: bool
    mask: np.ndarray  # Binary mask where 255 = watermark region
    confidence: float
    regions: list[tuple[int, int, int, int]]  # List of (x, y, w, h) bounding boxes
    watermark_type: WatermarkType = WatermarkType.NONE
    mask_coverage: float = 0.0  # Fraction of image covered by mask


class WatermarkDetector:
    """
    Detects watermark regions in images using multiple strategies.
    
    Strategies:
    1. Corner region analysis (watermarks often in corners)
    2. Semi-transparent overlay detection
    3. High-frequency text/pattern detection
    4. Known watermark template matching
    5. Repeating pattern detection (for tiled watermarks like INSTAHEADSHOTS)
    6. Full-image text detection
    """
    
    # Common watermark locations (as fractions of image dimensions)
    CORNER_REGIONS = [
        (0.0, 0.0, 0.25, 0.15),    # Top-left
        (0.75, 0.0, 0.25, 0.15),   # Top-right
        (0.0, 0.85, 0.25, 0.15),   # Bottom-left
        (0.75, 0.85, 0.25, 0.15),  # Bottom-right
        (0.35, 0.85, 0.30, 0.15),  # Bottom-center
    ]
    
    # Minimum confidence to consider a detection valid
    MIN_CONFIDENCE = 0.3
    
    # Known watermark text patterns (case-insensitive matching)
    KNOWN_WATERMARK_PATTERNS = [
        "instaheadshots",
        "aarzoo",
        "stock",
        "watermark",
        "preview",
        "sample",
        "shutterstock",
        "gettyimages",
        "adobe",
        "dreamstime",
    ]
    
    # Coverage threshold for classifying as repeating pattern
    # If mask covers more than this fraction, it's likely a repeating pattern
    REPEATING_PATTERN_COVERAGE_THRESHOLD = 0.25
    
    def __init__(self):
        """Initialize detector."""
        self._templates_loaded = False
        self._known_templates: list[np.ndarray] = []
        self._fft_analyzer = FFTPatternRemover()
        
    def _load_templates(self):
        """Load known watermark templates (lazy loading)."""
        if self._templates_loaded:
            return
            
        # Templates would be loaded from a templates directory
        # For now, we rely on heuristic detection
        self._templates_loaded = True
        logger.info("Watermark templates initialized")
        
    def _detect_corner_watermarks(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect watermarks in common corner locations.
        
        Returns mask and confidence score.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)
        max_confidence = 0.0
        
        for fx, fy, fw, fh in self.CORNER_REGIONS:
            # Calculate pixel coordinates
            x1 = int(fx * w)
            y1 = int(fy * h)
            x2 = int((fx + fw) * w)
            y2 = int((fy + fh) * h)
            
            region = gray[y1:y2, x1:x2]
            if region.size == 0:
                continue
                
            # Analyze region for watermark characteristics
            confidence = self._analyze_region(image[y1:y2, x1:x2], region)
            
            if confidence > self.MIN_CONFIDENCE:
                # Create mask for this region with some padding
                pad = 5
                mask[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)] = 255
                max_confidence = max(max_confidence, confidence)
                
        return mask, max_confidence
        
    def _analyze_region(self, color_region: np.ndarray, gray_region: np.ndarray) -> float:
        """
        Analyze a region to determine if it likely contains a watermark.
        
        Returns confidence score (0-1).
        """
        if color_region.size == 0 or gray_region.size == 0:
            return 0.0
            
        confidence = 0.0
        
        # Check 1: High-frequency content (text-like patterns)
        laplacian = cv2.Laplacian(gray_region, cv2.CV_64F)
        laplacian_var = laplacian.var()
        if laplacian_var > 500:  # High edge content
            confidence += 0.3
            
        # Check 2: Semi-transparency detection
        # Watermarks often have consistent semi-transparent values
        if len(color_region.shape) == 3:
            # Check for unusual color consistency that might indicate overlay
            color_std = np.std(color_region, axis=(0, 1))
            if np.mean(color_std) < 30:  # Low color variation
                confidence += 0.2
                
        # Check 3: Check for text-like structures using morphology
        _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Many small contours often indicate text
        small_contours = [c for c in contours if 10 < cv2.contourArea(c) < 500]
        if len(small_contours) > 5:
            confidence += 0.3
            
        # Check 4: Contrast with surrounding area
        mean_brightness = np.mean(gray_region)
        if mean_brightness > 200 or mean_brightness < 50:
            # Very bright or very dark regions might be watermarks
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def _detect_transparent_overlay(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect semi-transparent watermark overlays.
        
        Many AI tools add semi-transparent text or logos.
        """
        h, w = image.shape[:2]
        
        # Convert to LAB color space for better luminance analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Look for regions with unusual luminance patterns
        # Semi-transparent overlays often have consistent luminance shifts
        local_mean = cv2.blur(l_channel, (50, 50))
        diff = cv2.absdiff(l_channel, local_mean)
        
        # Threshold to find anomalous regions
        _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate confidence based on mask coverage
        coverage = np.sum(mask > 0) / mask.size
        
        # Too much or too little coverage suggests no watermark
        if 0.001 < coverage < 0.15:
            confidence = min(coverage * 10, 0.5)
        else:
            confidence = 0.0
            mask = np.zeros((h, w), dtype=np.uint8)
            
        return mask, confidence
        
    def _detect_edge_watermarks(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect watermarks along image edges.
        
        Some tools add watermarks as strips along edges.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)
        max_confidence = 0.0
        
        # Check bottom edge (most common)
        bottom_strip = gray[int(h * 0.9):, :]
        if bottom_strip.size > 0:
            # Check for uniform strip (banner-style watermark)
            std_dev = np.std(bottom_strip)
            if std_dev < 25:  # Very uniform
                mask[int(h * 0.9):, :] = 255
                max_confidence = 0.6
                
        return mask, max_confidence
    
    def _detect_repeating_pattern(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect repeating watermark patterns across the image.
        
        This handles tiled watermarks like "INSTAHEADSHOTS" that repeat
        diagonally or in a grid pattern across the entire image.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT to detect repeating patterns
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Log transform for better visualization
        magnitude_log = np.log1p(magnitude)
        
        # Find peaks in frequency domain (excluding DC component)
        center_y, center_x = h // 2, w // 2
        
        # Mask out the center (DC component and low frequencies)
        magnitude_log[center_y-10:center_y+10, center_x-10:center_x+10] = 0
        
        # Find strong frequency peaks that indicate repeating patterns
        threshold = np.percentile(magnitude_log, 99.5)
        peaks = magnitude_log > threshold
        
        # Count significant peaks
        num_peaks = np.sum(peaks)
        
        # If we have many peaks, it suggests a repeating pattern
        if num_peaks > 20:
            # Use edge detection to find the text/pattern
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to create regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours and filter by characteristics
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter for text-like contours (not too big, not too small)
                if 100 < area < (h * w * 0.1):
                    cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate confidence based on pattern regularity
            coverage = np.sum(mask > 0) / mask.size
            confidence = min(0.5 + (num_peaks / 100) * 0.3, 0.8) if 0.01 < coverage < 0.4 else 0.0
            
            return mask, confidence
        
        return np.zeros((h, w), dtype=np.uint8), 0.0
    
    def _detect_diagonal_lines(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect diagonal line patterns often used with watermarks.
        
        Many watermark services add thin diagonal lines across images.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply diagonal Sobel-like filters
        kernel_diag1 = np.array([[-1, 0, 1],
                                  [0, 0, 0],
                                  [1, 0, -1]], dtype=np.float32)
        kernel_diag2 = np.array([[1, 0, -1],
                                  [0, 0, 0],
                                  [-1, 0, 1]], dtype=np.float32)
        
        diag1 = cv2.filter2D(gray.astype(np.float32), -1, kernel_diag1)
        diag2 = cv2.filter2D(gray.astype(np.float32), -1, kernel_diag2)
        
        diag_combined = np.abs(diag1) + np.abs(diag2)
        
        # Threshold to find strong diagonal edges
        threshold = np.percentile(diag_combined, 95)
        lines_mask = (diag_combined > threshold).astype(np.uint8) * 255
        
        # Use Hough transform to confirm line presence
        lines = cv2.HoughLinesP(lines_mask, 1, np.pi/180, 50, 
                                 minLineLength=50, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            # Many diagonal lines detected - likely a watermark pattern
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Dilate the detected lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(lines_mask, kernel, iterations=2)
            
            confidence = min(0.3 + len(lines) / 100 * 0.4, 0.7)
            return mask, confidence
        
        return np.zeros((h, w), dtype=np.uint8), 0.0
    
    def _detect_text_everywhere(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect text-like structures across the entire image.
        
        Uses MSER (Maximally Stable Extremal Regions) for text detection.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create MSER detector
        mser = cv2.MSER_create()
        mser.setMinArea(50)
        mser.setMaxArea(5000)
        
        # Detect regions
        regions, _ = mser.detectRegions(gray)
        
        if len(regions) < 10:
            return np.zeros((h, w), dtype=np.uint8), 0.0
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Filter regions by aspect ratio (text-like)
        text_like_count = 0
        for region in regions:
            x, y, rw, rh = cv2.boundingRect(region)
            aspect = rw / max(rh, 1)
            
            # Text characters typically have aspect ratios between 0.2 and 5
            if 0.2 < aspect < 5:
                text_like_count += 1
                # Draw the region on mask
                hull = cv2.convexHull(region)
                cv2.fillPoly(mask, [hull], 255)
        
        # Dilate to connect nearby text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Calculate confidence based on distribution
        coverage = np.sum(mask > 0) / mask.size
        
        # If text is spread across image (not just in one corner), higher confidence
        # Check quadrant distribution
        q1 = np.sum(mask[:h//2, :w//2] > 0)
        q2 = np.sum(mask[:h//2, w//2:] > 0)
        q3 = np.sum(mask[h//2:, :w//2] > 0)
        q4 = np.sum(mask[h//2:, w//2:] > 0)
        
        total = q1 + q2 + q3 + q4
        if total > 0:
            min_quad = min(q1, q2, q3, q4) / total
            # Good distribution means watermark likely covers whole image
            distribution_score = min_quad * 4  # 0 to 1
        else:
            distribution_score = 0
        
        if 0.05 < coverage < 0.35 and text_like_count > 20:
            confidence = min(0.4 + distribution_score * 0.4, 0.8)
            return mask, confidence
        
        return np.zeros((h, w), dtype=np.uint8), 0.0
        
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect watermark regions in an image.
        
        Combines multiple detection strategies and returns combined result.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            DetectionResult with mask and confidence
        """
        self._load_templates()
        
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        confidences = []
        
        logger.debug(f"Running watermark detection on {w}x{h} image")
        
        # Strategy 1: Corner detection
        corner_mask, corner_conf = self._detect_corner_watermarks(image)
        if corner_conf > 0:
            combined_mask = cv2.bitwise_or(combined_mask, corner_mask)
            confidences.append(('corner', corner_conf))
            logger.debug(f"Corner detection: confidence={corner_conf:.2f}")
            
        # Strategy 2: Transparent overlay detection
        overlay_mask, overlay_conf = self._detect_transparent_overlay(image)
        if overlay_conf > 0:
            combined_mask = cv2.bitwise_or(combined_mask, overlay_mask)
            confidences.append(('overlay', overlay_conf))
            logger.debug(f"Overlay detection: confidence={overlay_conf:.2f}")
            
        # Strategy 3: Edge watermark detection
        edge_mask, edge_conf = self._detect_edge_watermarks(image)
        if edge_conf > 0:
            combined_mask = cv2.bitwise_or(combined_mask, edge_mask)
            confidences.append(('edge', edge_conf))
            logger.debug(f"Edge detection: confidence={edge_conf:.2f}")
        
        # Strategy 4: Repeating pattern detection (for tiled watermarks)
        pattern_mask, pattern_conf = self._detect_repeating_pattern(image)
        if pattern_conf > 0:
            combined_mask = cv2.bitwise_or(combined_mask, pattern_mask)
            confidences.append(('pattern', pattern_conf))
            logger.debug(f"Pattern detection: confidence={pattern_conf:.2f}")
        
        # Strategy 5: Diagonal line detection
        diag_mask, diag_conf = self._detect_diagonal_lines(image)
        if diag_conf > 0:
            combined_mask = cv2.bitwise_or(combined_mask, diag_mask)
            confidences.append(('diagonal', diag_conf))
            logger.debug(f"Diagonal detection: confidence={diag_conf:.2f}")
        
        # Strategy 6: Full-image text detection
        text_mask, text_conf = self._detect_text_everywhere(image)
        if text_conf > 0:
            combined_mask = cv2.bitwise_or(combined_mask, text_mask)
            confidences.append(('text', text_conf))
            logger.debug(f"Text detection: confidence={text_conf:.2f}")
            
        # Calculate mask coverage before dilation (for classification)
        raw_coverage = np.sum(combined_mask > 0) / combined_mask.size if combined_mask.size > 0 else 0.0
        
        # Dilate mask slightly for better inpainting coverage
        if np.sum(combined_mask > 0) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
            
        # Calculate overall confidence (max of all strategies)
        overall_confidence = max([c[1] for c in confidences]) if confidences else 0.0
        detected = overall_confidence >= self.MIN_CONFIDENCE
        
        if confidences:
            best_strategy = max(confidences, key=lambda x: x[1])
            logger.info(f"Best detection: {best_strategy[0]} with confidence {best_strategy[1]:.2f}")
        
        # Find bounding boxes of detected regions
        regions = []
        if detected:
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                regions.append((x, y, cw, ch))
        
        # Calculate final mask coverage
        mask_coverage = np.sum(combined_mask > 0) / combined_mask.size if combined_mask.size > 0 else 0.0
        
        # Classify watermark type
        watermark_type = self._classify_watermark_type(image, mask_coverage, confidences)
        
        if detected:
            logger.info(f"Detected {len(regions)} watermark regions, mask coverage: {mask_coverage:.1%}, type: {watermark_type.value}")
                
        return DetectionResult(
            detected=detected,
            mask=combined_mask,
            confidence=overall_confidence,
            regions=regions,
            watermark_type=watermark_type,
            mask_coverage=mask_coverage
        )
    
    def _classify_watermark_type(
        self, 
        image: np.ndarray, 
        mask_coverage: float,
        confidences: list[tuple[str, float]]
    ) -> WatermarkType:
        """
        Classify the type of watermark to determine best removal strategy.
        
        Args:
            image: Original BGR image
            mask_coverage: Fraction of image covered by detection mask
            confidences: List of (strategy_name, confidence) tuples
            
        Returns:
            WatermarkType indicating best removal approach
        """
        if not confidences:
            return WatermarkType.NONE
            
        # High coverage strongly suggests repeating pattern
        if mask_coverage > self.REPEATING_PATTERN_COVERAGE_THRESHOLD:
            # Verify with FFT analysis
            pattern_analysis = self._fft_analyzer.analyze_pattern(image)
            
            if pattern_analysis.has_repeating_pattern:
                logger.info(f"Classified as REPEATING_PATTERN (coverage: {mask_coverage:.1%}, "
                           f"pattern strength: {pattern_analysis.pattern_strength:.2f})")
                return WatermarkType.REPEATING_PATTERN
            else:
                # High coverage but no clear pattern - might still be better with FFT
                # if the pattern detection strategies (not corner/edge) are dominant
                pattern_strategies = ['pattern', 'diagonal', 'text']
                pattern_conf = sum(c[1] for c in confidences if c[0] in pattern_strategies)
                localized_strategies = ['corner', 'edge', 'overlay']
                localized_conf = sum(c[1] for c in confidences if c[0] in localized_strategies)
                
                if pattern_conf > localized_conf:
                    logger.info(f"Classified as REPEATING_PATTERN based on detection strategies")
                    return WatermarkType.REPEATING_PATTERN
        
        # Lower coverage or no pattern - use inpainting
        if mask_coverage > 0:
            logger.info(f"Classified as LOCALIZED (coverage: {mask_coverage:.1%})")
            return WatermarkType.LOCALIZED
            
        return WatermarkType.NONE


class WatermarkRemover:
    """
    Removes watermarks from images using the best strategy for each type:
    
    - LOCALIZED watermarks (corners, edges, banners): LaMa inpainting
    - REPEATING_PATTERN watermarks (tiled text like INSTAHEADSHOTS): SLBR neural network
    
    LaMa inpainting is optimized for small masked regions.
    SLBR is a neural network specifically trained for visible watermark removal.
    """
    
    def __init__(self):
        """Initialize remover with lazy model loading."""
        self._torch_model = None
        self._model_loaded = False
        self._device = 'cpu'
        self._detector = WatermarkDetector()
        self._fft_remover = FFTPatternRemover()
        self._slbr_remover = None  # Lazy load
        
    def _load_model(self):
        """Load LaMa model lazily."""
        if self._model_loaded:
            return
            
        logger.info("Loading LaMa inpainting model...")
        
        try:
            import torch
            from simple_lama_inpainting.models.model import download_model, LAMA_MODEL_URL
            
            # Download/get cached model path
            model_path = download_model(LAMA_MODEL_URL)
            
            # Load with map_location to handle CUDA-saved models on non-CUDA devices
            device = get_device()
            if device == 'mps':
                # MPS doesn't work with this TorchScript model, use CPU
                map_location = 'cpu'
                self._device = 'cpu'
            elif device == 'cuda':
                map_location = 'cuda'
                self._device = 'cuda'
            else:
                map_location = 'cpu'
                self._device = 'cpu'
            
            logger.info(f"Loading LaMa model with map_location={map_location}")
            self._torch_model = torch.jit.load(model_path, map_location=map_location)
            self._torch_model.eval()
            self._model_loaded = True
            
            logger.info(f"LaMa model loaded on {self._device}")
            
        except ImportError:
            logger.error(
                "simple-lama-inpainting not installed. "
                "Install with: pip install simple-lama-inpainting"
            )
            raise
            
    def _inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Inpaint the masked region using LaMa.
        
        Args:
            image: PIL Image to process
            mask: Binary mask where 255 = region to inpaint
            
        Returns:
            Inpainted PIL Image
        """
        import torch
        import torch.nn.functional as F
        
        self._load_model()
        
        # Prepare image tensor
        img_np = np.array(image)
        orig_h, orig_w = img_np.shape[:2]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Prepare mask tensor (normalize to 0-1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        # LaMa requires dimensions divisible by 8
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            # Pad image and mask (right and bottom padding)
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Move to device
        img_tensor = img_tensor.to(self._device)
        mask_tensor = mask_tensor.to(self._device)
        
        # Run inpainting
        with torch.no_grad():
            result = self._torch_model(img_tensor, mask_tensor)
        
        # Remove padding if applied
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :orig_h, :orig_w]
        
        # Convert back to PIL
        result_np = (result[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        result_pil = Image.fromarray(result_np)
        
        return result_pil
        
    def remove(
        self,
        image_path: Path,
        custom_mask: Optional[np.ndarray] = None,
        force_method: Optional[str] = None,
    ) -> WatermarkResult:
        """
        Remove watermark from a single image.
        
        Automatically selects the best removal strategy:
        - FFT filtering for repeating patterns (preserves underlying content)
        - LaMa inpainting for localized watermarks (regenerates masked regions)
        
        Args:
            image_path: Path to image file
            custom_mask: Optional custom mask (if None, auto-detect)
            force_method: Optional override - "fft" or "inpaint"
            
        Returns:
            WatermarkResult with cleaned image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        pil_image = Image.open(image_path).convert("RGB")
        
        # Detect or use custom mask
        if custom_mask is not None:
            mask = custom_mask
            detection = DetectionResult(
                detected=True,
                mask=mask,
                confidence=1.0,
                regions=[],
                watermark_type=WatermarkType.LOCALIZED,
                mask_coverage=np.sum(mask > 0) / mask.size if mask.size > 0 else 0.0
            )
        else:
            detection = self._detector.detect(image)
            mask = detection.mask
            
        # If no watermark detected, return original
        if not detection.detected or np.sum(mask > 0) == 0:
            return WatermarkResult(
                original_path=image_path,
                cleaned_image=pil_image,
                watermark_detected=False,
                confidence=detection.confidence,
                watermark_type=WatermarkType.NONE,
                method_used=""
            )
        
        # Determine removal method
        if force_method == "fft":
            use_fft = True
        elif force_method == "inpaint":
            use_fft = False
        else:
            # Auto-select based on watermark type
            use_fft = detection.watermark_type == WatermarkType.REPEATING_PATTERN
        
        # Apply appropriate removal method
        if use_fft:
            # Try SLBR neural network first (better for complex watermarks)
            try:
                from .slbr_remover import SLBRWatermarkRemover, is_slbr_available
                
                if is_slbr_available():
                    if self._slbr_remover is None:
                        logger.info("Loading SLBR neural network for watermark removal...")
                        self._slbr_remover = SLBRWatermarkRemover()
                    
                    # Use enhanced mode for aggressive watermarks (high coverage)
                    if detection.mask_coverage > 0.4:
                        logger.info("Using SLBR enhanced mode (double pass) for aggressive watermark")
                        cleaned_cv = self._slbr_remover.remove_watermark_enhanced(image)
                    else:
                        logger.info("Using SLBR neural network for watermark removal")
                        cleaned_cv = self._slbr_remover.remove_watermark(
                            image, 
                            process_size=1024,
                            num_passes=1,
                            remove_banner=True,
                            apply_smoothing=True
                        )
                    cleaned = Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))
                    method_used = "slbr"
                else:
                    # Fall back to FFT if SLBR not available
                    logger.info("SLBR not available, using FFT filtering")
                    cleaned_cv = self._fft_remover.remove_pattern(image)
                    cleaned = Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))
                    method_used = "fft"
            except Exception as e:
                logger.warning(f"SLBR failed ({e}), falling back to FFT filtering")
                cleaned_cv = self._fft_remover.remove_pattern(image)
                cleaned = Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))
                method_used = "fft"
        else:
            logger.info(f"Using LaMa inpainting for localized watermark removal")
            cleaned = self._inpaint(pil_image, mask)
            method_used = "inpaint"
        
        return WatermarkResult(
            original_path=image_path,
            cleaned_image=cleaned,
            watermark_detected=True,
            mask_used=mask,
            confidence=detection.confidence,
            watermark_type=detection.watermark_type,
            method_used=method_used
        )
        
    def remove_from_pil(
        self,
        image: Image.Image,
        custom_mask: Optional[np.ndarray] = None,
        force_method: Optional[str] = None,
    ) -> tuple[Image.Image, bool, float, str]:
        """
        Remove watermark from a PIL Image directly.
        
        Args:
            image: PIL Image to process
            custom_mask: Optional custom mask
            force_method: Optional override - "fft" or "inpaint"
            
        Returns:
            Tuple of (cleaned_image, watermark_detected, confidence, method_used)
        """
        # Convert to OpenCV format for detection
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect or use custom mask
        if custom_mask is not None:
            mask = custom_mask
            detected = True
            confidence = 1.0
            watermark_type = WatermarkType.LOCALIZED
        else:
            detection = self._detector.detect(cv_image)
            mask = detection.mask
            detected = detection.detected
            confidence = detection.confidence
            watermark_type = detection.watermark_type
            
        # If no watermark detected, return original
        if not detected or np.sum(mask > 0) == 0:
            return image, False, confidence, ""
        
        # Determine removal method
        if force_method == "fft":
            use_fft = True
        elif force_method == "inpaint":
            use_fft = False
        else:
            use_fft = watermark_type == WatermarkType.REPEATING_PATTERN
        
        # Apply appropriate removal method
        if use_fft:
            # Try SLBR neural network first (better for complex watermarks)
            try:
                from .slbr_remover import SLBRWatermarkRemover, is_slbr_available
                
                if is_slbr_available():
                    if self._slbr_remover is None:
                        logger.info("Loading SLBR neural network for watermark removal...")
                        self._slbr_remover = SLBRWatermarkRemover()
                    
                    logger.info("Using SLBR neural network for watermark removal")
                    cleaned_cv = self._slbr_remover.remove_watermark(cv_image, process_size=512)
                    cleaned = Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))
                    method_used = "slbr"
                else:
                    logger.info("SLBR not available, using FFT filtering")
                    cleaned_cv = self._fft_remover.remove_pattern(cv_image)
                    cleaned = Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))
                    method_used = "fft"
            except Exception as e:
                logger.warning(f"SLBR failed ({e}), falling back to FFT filtering")
                cleaned_cv = self._fft_remover.remove_pattern(cv_image)
                cleaned = Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))
                method_used = "fft"
        else:
            logger.info(f"Using LaMa inpainting for localized watermark removal")
            cleaned = self._inpaint(image, mask)
            method_used = "inpaint"
        
        return cleaned, True, confidence, method_used
        
    def remove_batch(
        self,
        image_paths: list[Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[WatermarkResult]:
        """
        Remove watermarks from multiple images.
        
        Args:
            image_paths: List of image paths to process
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            List of WatermarkResults
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, total, f"Processing {path.name}")
                
            try:
                result = self.remove(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                # Return original on error
                try:
                    pil_image = Image.open(path).convert("RGB")
                    results.append(WatermarkResult(
                        original_path=path,
                        cleaned_image=pil_image,
                        watermark_detected=False,
                        confidence=0.0
                    ))
                except Exception:
                    logger.error(f"Could not even load {path}")
                    
        return results
        
    def cleanup(self):
        """Release model resources."""
        self._torch_model = None
        self._model_loaded = False
        if self._slbr_remover is not None:
            self._slbr_remover.cleanup()
            self._slbr_remover = None
        logger.info("WatermarkRemover resources released")
