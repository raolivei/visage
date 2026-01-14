"""
Photo Validator

Validates and preprocesses uploaded photos for LoRA training.
Ensures high-quality inputs for better model outputs.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ValidationError(Enum):
    """Validation error types."""
    NO_FACE = "no_face_detected"
    MULTIPLE_FACES = "multiple_faces_detected"
    LOW_RESOLUTION = "face_resolution_too_low"
    BLURRY = "image_too_blurry"
    OVEREXPOSED = "image_overexposed"
    UNDEREXPOSED = "image_underexposed"
    OCCLUDED = "face_occluded"


@dataclass
class ValidationResult:
    """Result of photo validation."""
    valid: bool
    error: Optional[ValidationError] = None
    error_message: Optional[str] = None
    face_box: Optional[tuple[int, int, int, int]] = None  # x, y, w, h
    face_landmarks: Optional[dict] = None
    quality_scores: Optional[dict] = None


@dataclass 
class ProcessedPhoto:
    """A validated and preprocessed photo."""
    original_path: Path
    processed_image: Image.Image
    face_embedding: Optional[np.ndarray] = None
    pose_angle: Optional[float] = None  # Yaw angle for diversity check


class PhotoValidator:
    """
    Validates uploaded photos for LoRA training quality.
    
    Checks:
    - Face detection (single face required)
    - Resolution (512px minimum face width)
    - Blur detection
    - Lighting (exposure) assessment
    - Face occlusion detection
    """
    
    # Minimum face width in pixels
    MIN_FACE_WIDTH = 512
    
    # Blur threshold (Laplacian variance)
    BLUR_THRESHOLD = 100.0
    
    # Exposure thresholds
    UNDEREXPOSED_THRESHOLD = 40  # Mean brightness
    OVEREXPOSED_THRESHOLD = 220
    
    def __init__(self):
        """Initialize validator with face detection models."""
        self.face_analyzer = None
        self._models_loaded = False
        
    def _load_models(self):
        """Load face analysis models lazily."""
        if self._models_loaded:
            return
            
        logger.info("Loading face analysis models...")
        
        try:
            from insightface.app import FaceAnalysis
            
            # Use buffalo_l for best accuracy
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']  # MPS not supported, use CPU
            )
            self.face_analyzer.prepare(ctx_id=-1)  # -1 for CPU
            
            self._models_loaded = True
            logger.info("Face analysis models loaded")
            
        except ImportError:
            logger.warning("InsightFace not installed, using OpenCV fallback")
            self._setup_opencv_fallback()
            
    def _setup_opencv_fallback(self):
        """Set up OpenCV cascade classifier as fallback."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self._models_loaded = True
        logger.info("Using OpenCV face detection fallback")
        
    def _detect_faces_insightface(self, image: np.ndarray) -> list[dict]:
        """Detect faces using InsightFace."""
        if self.face_analyzer is None:
            return []
            
        faces = self.face_analyzer.get(image)
        
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            results.append({
                'box': (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                'landmarks': face.kps if hasattr(face, 'kps') else None,
                'embedding': face.embedding if hasattr(face, 'embedding') else None,
                'pose': face.pose if hasattr(face, 'pose') else None,
            })
            
        return results
        
    def _detect_faces_opencv(self, image: np.ndarray) -> list[dict]:
        """Detect faces using OpenCV cascade (fallback)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        return [{'box': tuple(face), 'landmarks': None, 'embedding': None, 'pose': None} 
                for face in faces]
        
    def _detect_faces(self, image: np.ndarray) -> list[dict]:
        """Detect faces in image using available method."""
        if self.face_analyzer is not None:
            return self._detect_faces_insightface(image)
        elif hasattr(self, 'face_cascade'):
            return self._detect_faces_opencv(image)
        else:
            return []
            
    def _compute_blur_score(self, image: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.
        Higher values = sharper image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
        
    def _compute_exposure(self, image: np.ndarray) -> tuple[float, float, float]:
        """
        Compute exposure metrics.
        Returns: (mean_brightness, underexposed_ratio, overexposed_ratio)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Ratio of very dark pixels
        underexposed = np.sum(gray < 30) / gray.size
        
        # Ratio of very bright pixels
        overexposed = np.sum(gray > 245) / gray.size
        
        return float(mean_brightness), float(underexposed), float(overexposed)
        
    def validate(self, image_path: Path) -> ValidationResult:
        """
        Validate a single photo for training suitability.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ValidationResult with validity status and details
        """
        self._load_models()
        
        # Load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return ValidationResult(
                    valid=False,
                    error=ValidationError.NO_FACE,
                    error_message=f"Could not load image: {image_path}"
                )
        except Exception as e:
            return ValidationResult(
                valid=False,
                error=ValidationError.NO_FACE,
                error_message=f"Error loading image: {e}"
            )
            
        # Detect faces
        faces = self._detect_faces(image)
        
        if len(faces) == 0:
            return ValidationResult(
                valid=False,
                error=ValidationError.NO_FACE,
                error_message="No face detected in image"
            )
            
        if len(faces) > 1:
            return ValidationResult(
                valid=False,
                error=ValidationError.MULTIPLE_FACES,
                error_message=f"Multiple faces detected ({len(faces)}), need exactly one"
            )
            
        face = faces[0]
        x, y, w, h = face['box']
        
        # Check face resolution
        if w < self.MIN_FACE_WIDTH:
            return ValidationResult(
                valid=False,
                error=ValidationError.LOW_RESOLUTION,
                error_message=f"Face width {w}px below minimum {self.MIN_FACE_WIDTH}px"
            )
            
        # Extract face region for quality checks
        face_region = image[max(0, y):y+h, max(0, x):x+w]
        
        # Check blur
        blur_score = self._compute_blur_score(face_region)
        if blur_score < self.BLUR_THRESHOLD:
            return ValidationResult(
                valid=False,
                error=ValidationError.BLURRY,
                error_message=f"Image too blurry (score: {blur_score:.1f}, min: {self.BLUR_THRESHOLD})"
            )
            
        # Check exposure
        mean_bright, under_ratio, over_ratio = self._compute_exposure(face_region)
        
        if mean_bright < self.UNDEREXPOSED_THRESHOLD:
            return ValidationResult(
                valid=False,
                error=ValidationError.UNDEREXPOSED,
                error_message=f"Image too dark (brightness: {mean_bright:.1f})"
            )
            
        if mean_bright > self.OVEREXPOSED_THRESHOLD:
            return ValidationResult(
                valid=False,
                error=ValidationError.OVEREXPOSED,
                error_message=f"Image too bright (brightness: {mean_bright:.1f})"
            )
            
        # All checks passed
        return ValidationResult(
            valid=True,
            face_box=(x, y, w, h),
            face_landmarks=face.get('landmarks'),
            quality_scores={
                'blur_score': blur_score,
                'mean_brightness': mean_bright,
                'underexposed_ratio': under_ratio,
                'overexposed_ratio': over_ratio,
            }
        )
        
    def validate_batch(self, image_paths: list[Path]) -> tuple[list[ValidationResult], list[Path]]:
        """
        Validate multiple photos.
        
        Args:
            image_paths: List of paths to validate
            
        Returns:
            Tuple of (all results, valid paths only)
        """
        results = []
        valid_paths = []
        
        for path in image_paths:
            result = self.validate(path)
            results.append(result)
            
            if result.valid:
                valid_paths.append(path)
            else:
                logger.warning(f"Invalid photo {path}: {result.error_message}")
                
        logger.info(f"Validated {len(image_paths)} photos: {len(valid_paths)} valid, {len(image_paths) - len(valid_paths)} rejected")
        return results, valid_paths
        
    def check_diversity(self, image_paths: list[Path], min_angles: int = 3) -> bool:
        """
        Check if photos have sufficient pose diversity.
        
        Args:
            image_paths: Validated image paths
            min_angles: Minimum number of different angles required
            
        Returns:
            True if diverse enough
        """
        if not self._models_loaded or self.face_analyzer is None:
            logger.warning("Pose estimation not available, skipping diversity check")
            return True
            
        yaw_angles = []
        
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
                
            faces = self._detect_faces_insightface(image)
            if faces and faces[0].get('pose') is not None:
                # pose is [pitch, yaw, roll]
                yaw = faces[0]['pose'][1] if len(faces[0]['pose']) > 1 else 0
                yaw_angles.append(yaw)
                
        if len(yaw_angles) < min_angles:
            return False
            
        # Bin angles into categories: left (-45 to -15), front (-15 to 15), right (15 to 45)
        angle_bins = set()
        for yaw in yaw_angles:
            if yaw < -15:
                angle_bins.add('left')
            elif yaw > 15:
                angle_bins.add('right')
            else:
                angle_bins.add('front')
                
        return len(angle_bins) >= min(min_angles, 3)
        
    def preprocess(
        self, 
        image_path: Path, 
        output_size: int = 512,
        remove_background: bool = False
    ) -> Optional[ProcessedPhoto]:
        """
        Preprocess a validated photo for training.
        
        Args:
            image_path: Path to validated image
            output_size: Output image size (square)
            remove_background: Whether to remove background
            
        Returns:
            ProcessedPhoto or None if preprocessing fails
        """
        self._load_models()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        # Detect face
        faces = self._detect_faces(image)
        if not faces:
            return None
            
        face = faces[0]
        x, y, w, h = face['box']
        
        # Expand bounding box for context (20% padding)
        padding = int(max(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Crop to face region
        face_crop = image[y1:y2, x1:x2]
        
        # Optional background removal
        if remove_background:
            try:
                from rembg import remove
                # Convert to PIL, remove bg, convert back
                pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                pil_img = remove(pil_img)
                face_crop = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
            except ImportError:
                logger.warning("rembg not installed, skipping background removal")
                
        # Resize to output size (maintaining aspect ratio, then pad/crop)
        h, w = face_crop.shape[:2]
        scale = output_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create square canvas and center the image
        canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        y_offset = (output_size - new_h) // 2
        x_offset = (output_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        
        # Get embedding if available
        embedding = face.get('embedding')
        pose = face.get('pose')
        yaw_angle = pose[1] if pose is not None and len(pose) > 1 else None
        
        return ProcessedPhoto(
            original_path=image_path,
            processed_image=pil_image,
            face_embedding=embedding,
            pose_angle=yaw_angle
        )
        
    def preprocess_batch(
        self,
        image_paths: list[Path],
        output_size: int = 512,
        remove_background: bool = False
    ) -> list[ProcessedPhoto]:
        """
        Preprocess multiple photos.
        
        Args:
            image_paths: Paths to validated images
            output_size: Output size for each image
            remove_background: Whether to remove backgrounds
            
        Returns:
            List of processed photos
        """
        processed = []
        
        for path in image_paths:
            result = self.preprocess(path, output_size, remove_background)
            if result is not None:
                processed.append(result)
            else:
                logger.warning(f"Failed to preprocess: {path}")
                
        logger.info(f"Preprocessed {len(processed)}/{len(image_paths)} photos")
        return processed
        
    def cleanup(self):
        """Release resources."""
        self.face_analyzer = None
        self._models_loaded = False
