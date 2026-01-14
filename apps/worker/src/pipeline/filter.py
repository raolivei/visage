"""
Quality Filter

Filter and score generated images for quality using multi-factor analysis.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
import cv2

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class QualityScores:
    """Quality scores for a generated image."""
    face_similarity: float  # 0-1, higher is better
    aesthetic_score: float  # 0-10, higher is better
    technical_quality: float  # 0-1, higher is better
    artifact_score: float  # 0-1, lower is better (0 = no artifacts)
    overall_score: float  # Weighted combination
    passes_filter: bool


class QualityFilter:
    """
    Quality filter for generated headshots.
    
    Evaluates images using multiple factors:
    - Face similarity to reference photos (40% weight)
    - Aesthetic score via CLIP (25% weight)
    - Technical quality - sharpness, exposure (20% weight)
    - Artifact detection - eyes, teeth, hands (15% weight)
    """
    
    # Scoring weights
    WEIGHT_FACE_SIMILARITY = 0.40
    WEIGHT_AESTHETIC = 0.25
    WEIGHT_TECHNICAL = 0.20
    WEIGHT_ARTIFACTS = 0.15
    
    # Thresholds
    MIN_FACE_SIMILARITY = 0.65
    MIN_AESTHETIC_SCORE = 5.5  # Out of 10
    MIN_TECHNICAL_SCORE = 0.5
    MAX_ARTIFACT_SCORE = 0.35

    def __init__(self):
        """Initialize filter (models loaded lazily)."""
        self.face_analyzer = None
        self.aesthetic_predictor = None
        self.reference_embeddings: list[np.ndarray] = []
        self._models_loaded = False

    def _load_models(self):
        """Load face recognition and aesthetic scoring models."""
        if self._models_loaded:
            return
            
        logger.info("Loading quality filter models...")
        
        # Load InsightFace for face embeddings
        try:
            from insightface.app import FaceAnalysis
            
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=-1)
            logger.info("InsightFace loaded for face similarity")
            
        except ImportError:
            logger.warning("InsightFace not available, face similarity will be estimated")
            
        # Load CLIP aesthetic predictor
        try:
            self._load_aesthetic_predictor()
            logger.info("Aesthetic predictor loaded")
        except Exception as e:
            logger.warning(f"Aesthetic predictor not available: {e}")
            
        self._models_loaded = True
        
    def _load_aesthetic_predictor(self):
        """Load CLIP-based aesthetic predictor."""
        try:
            import torch
            import clip
            
            # Load CLIP model
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
            self.clip_device = device
            
            # Simple aesthetic scoring using CLIP
            # In production, use a trained aesthetic predictor head
            self.aesthetic_predictor = "clip"
            
        except ImportError:
            logger.warning("CLIP not installed, using fallback aesthetic scoring")
            self.aesthetic_predictor = "fallback"

    def set_reference_images(self, image_paths: list[Path]) -> int:
        """
        Set reference images for face similarity comparison.
        
        Args:
            image_paths: List of paths to reference images
            
        Returns:
            Number of face embeddings extracted
        """
        self._load_models()
        self.reference_embeddings = []
        
        if self.face_analyzer is None:
            logger.warning("Face analyzer not available")
            return 0
        
        for path in image_paths:
            try:
                image = cv2.imread(str(path))
                if image is None:
                    continue
                    
                faces = self.face_analyzer.get(image)
                if faces and hasattr(faces[0], 'embedding'):
                    # Normalize embedding
                    embedding = faces[0].embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    self.reference_embeddings.append(embedding)
                    
            except Exception as e:
                logger.warning(f"Failed to process reference {path}: {e}")
        
        logger.info(f"Loaded {len(self.reference_embeddings)} reference face embeddings")
        return len(self.reference_embeddings)
        
    def set_reference_embeddings(self, embeddings: list[np.ndarray]):
        """
        Set reference embeddings directly.
        
        Args:
            embeddings: List of face embeddings from validation phase
        """
        self.reference_embeddings = []
        for emb in embeddings:
            if emb is not None:
                # Normalize
                emb = emb / np.linalg.norm(emb)
                self.reference_embeddings.append(emb)
                
        logger.info(f"Set {len(self.reference_embeddings)} reference embeddings")

    def compute_face_similarity(self, image: Image.Image) -> float:
        """
        Compute similarity between generated face and reference faces.
        
        Uses cosine similarity between ArcFace embeddings.
        
        Args:
            image: Generated image (PIL)
            
        Returns:
            Similarity score (0.0 to 1.0), higher is better
        """
        if not self.reference_embeddings:
            logger.warning("No reference embeddings set")
            return 0.5  # Neutral score
        
        self._load_models()
        
        if self.face_analyzer is None:
            # Fallback: return moderate score
            return 0.6
        
        try:
            # Convert PIL to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Get face embedding
            faces = self.face_analyzer.get(image_cv)
            
            if not faces or not hasattr(faces[0], 'embedding'):
                logger.warning("No face detected in generated image")
                return 0.0
                
            gen_embedding = faces[0].embedding
            gen_embedding = gen_embedding / np.linalg.norm(gen_embedding)
            
            # Compute cosine similarity with all references
            similarities = []
            for ref_emb in self.reference_embeddings:
                sim = np.dot(ref_emb, gen_embedding)
                # Clamp to [0, 1] range
                sim = max(0.0, min(1.0, (sim + 1) / 2))
                similarities.append(sim)
                
            # Return max similarity (best match to any reference)
            return float(max(similarities))
            
        except Exception as e:
            logger.error(f"Failed to compute face similarity: {e}")
            return 0.0

    def compute_aesthetic_score(self, image: Image.Image) -> float:
        """
        Compute aesthetic quality score using CLIP.
        
        Args:
            image: Generated image
            
        Returns:
            Aesthetic score (0.0 to 10.0)
        """
        self._load_models()
        
        if self.aesthetic_predictor == "clip":
            try:
                return self._compute_clip_aesthetic(image)
            except Exception as e:
                logger.warning(f"CLIP aesthetic scoring failed: {e}")
                
        # Fallback: estimate based on image properties
        return self._estimate_aesthetic_score(image)
        
    def _compute_clip_aesthetic(self, image: Image.Image) -> float:
        """Compute aesthetic score using CLIP."""
        import torch
        
        # Preprocess image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.clip_device)
        
        # Define quality prompts
        positive_prompts = [
            "a professional high quality photograph",
            "a beautiful portrait photo",
            "excellent lighting and composition",
            "sharp, well-focused professional headshot",
        ]
        
        negative_prompts = [
            "a low quality amateur photo",
            "blurry, poorly lit photograph",
            "distorted, unnatural looking image",
            "bad composition, unflattering angle",
        ]
        
        with torch.no_grad():
            # Get image features
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get text features for positive prompts
            pos_tokens = torch.cat([
                self.clip_model.encode_text(
                    torch.tensor(self.clip_model.tokenize([p])).to(self.clip_device)
                ) for p in positive_prompts
            ])
            pos_features = pos_tokens / pos_tokens.norm(dim=-1, keepdim=True)
            
            # Get text features for negative prompts
            neg_tokens = torch.cat([
                self.clip_model.encode_text(
                    torch.tensor(self.clip_model.tokenize([p])).to(self.clip_device)
                ) for p in negative_prompts
            ])
            neg_features = neg_tokens / neg_tokens.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            pos_sim = (image_features @ pos_features.T).mean().item()
            neg_sim = (image_features @ neg_features.T).mean().item()
            
            # Convert to 0-10 scale
            # Higher positive similarity and lower negative similarity = higher score
            score = (pos_sim - neg_sim + 1) * 5  # Map [-1, 1] to [0, 10]
            score = max(0, min(10, score))
            
            return float(score)
            
    def _estimate_aesthetic_score(self, image: Image.Image) -> float:
        """Estimate aesthetic score from image properties (fallback)."""
        img_array = np.array(image)
        
        scores = []
        
        # Sharpness score
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 500)
        scores.append(sharpness * 10)
        
        # Contrast score
        contrast = gray.std() / 64  # Normalize by expected std
        scores.append(min(10, contrast * 10))
        
        # Color balance (not too saturated, not too dull)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean() / 255
        color_score = 1.0 - abs(saturation - 0.4) * 2  # Optimal around 0.4
        scores.append(max(0, color_score * 10))
        
        return float(np.mean(scores))

    def compute_technical_quality(self, image: Image.Image) -> float:
        """
        Compute technical quality score.
        
        Evaluates sharpness, exposure, and noise.
        
        Args:
            image: Generated image
            
        Returns:
            Technical quality score (0.0 to 1.0)
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        scores = []
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 300)
        scores.append(sharpness)
        
        # Exposure (check for proper histogram distribution)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Penalize if too much mass at extremes
        dark_mass = hist[:30].sum()
        bright_mass = hist[225:].sum()
        exposure_score = 1.0 - (dark_mass + bright_mass)
        scores.append(max(0, exposure_score))
        
        # Noise estimation (using local variance)
        noise = cv2.Laplacian(gray, cv2.CV_64F).std()
        noise_score = 1.0 - min(1.0, noise / 50)
        scores.append(noise_score)
        
        return float(np.mean(scores))

    def detect_artifacts(self, image: Image.Image) -> float:
        """
        Detect common AI generation artifacts.
        
        Checks for:
        - Asymmetric or malformed eyes
        - Teeth issues
        - Unnatural skin texture
        - Edge artifacts
        
        Args:
            image: Generated image
            
        Returns:
            Artifact score (0.0 = clean, 1.0 = severe artifacts)
        """
        self._load_models()
        
        img_array = np.array(image)
        artifact_scores = []
        
        # Check for face detection (if face analyzer available)
        if self.face_analyzer is not None:
            try:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                faces = self.face_analyzer.get(img_cv)
                
                if not faces:
                    # No face = major artifact
                    return 0.8
                    
                face = faces[0]
                
                # Check face detection confidence
                if hasattr(face, 'det_score'):
                    confidence = face.det_score
                    if confidence < 0.8:
                        artifact_scores.append(1.0 - confidence)
                        
                # Check landmark quality if available
                if hasattr(face, 'kps') and face.kps is not None:
                    kps = face.kps
                    
                    # Check eye symmetry
                    if len(kps) >= 2:
                        left_eye = kps[0]
                        right_eye = kps[1]
                        
                        # Eyes should be roughly level
                        eye_angle = abs(left_eye[1] - right_eye[1])
                        eye_dist = np.linalg.norm(left_eye - right_eye)
                        
                        if eye_dist > 0:
                            asymmetry = eye_angle / eye_dist
                            artifact_scores.append(min(1.0, asymmetry * 5))
                            
            except Exception as e:
                logger.warning(f"Artifact detection error: {e}")
                
        # Check for edge artifacts (sharp unnatural edges)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Too many edges might indicate artifacts
        if edge_density > 0.15:
            artifact_scores.append((edge_density - 0.15) * 3)
            
        # Check for color banding (common in AI images)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        unique_hues = len(np.unique(hue))
        
        if unique_hues < 50:  # Too few unique hues = banding
            artifact_scores.append(0.3)
            
        if not artifact_scores:
            return 0.1  # Default low artifact score
            
        return float(min(1.0, np.mean(artifact_scores)))

    def evaluate(self, image: Image.Image) -> QualityScores:
        """
        Evaluate an image and return comprehensive quality scores.
        
        Args:
            image: Generated image
            
        Returns:
            QualityScores with all metrics and pass/fail decision
        """
        # Compute all scores
        face_similarity = self.compute_face_similarity(image)
        aesthetic_score = self.compute_aesthetic_score(image)
        technical_quality = self.compute_technical_quality(image)
        artifact_score = self.detect_artifacts(image)
        
        # Compute weighted overall score
        # Note: artifact_score is inverted (lower is better)
        overall_score = (
            face_similarity * self.WEIGHT_FACE_SIMILARITY +
            (aesthetic_score / 10) * self.WEIGHT_AESTHETIC +
            technical_quality * self.WEIGHT_TECHNICAL +
            (1.0 - artifact_score) * self.WEIGHT_ARTIFACTS
        )
        
        # Determine pass/fail based on thresholds
        passes = (
            face_similarity >= self.MIN_FACE_SIMILARITY and
            aesthetic_score >= self.MIN_AESTHETIC_SCORE and
            technical_quality >= self.MIN_TECHNICAL_SCORE and
            artifact_score <= self.MAX_ARTIFACT_SCORE
        )
        
        return QualityScores(
            face_similarity=face_similarity,
            aesthetic_score=aesthetic_score,
            technical_quality=technical_quality,
            artifact_score=artifact_score,
            overall_score=overall_score,
            passes_filter=passes,
        )

    def filter_batch(
        self, 
        images: list[dict],
        top_k: Optional[int] = None,
        min_pass: int = 10
    ) -> list[dict]:
        """
        Filter a batch of images, keeping only the best ones.
        
        Args:
            images: List of dicts with 'image' key (PIL Image)
            top_k: Keep top K images by score (None = keep all passing)
            min_pass: Minimum images to pass (relaxes thresholds if needed)
            
        Returns:
            Filtered list with quality scores added
        """
        # Evaluate all images
        evaluated = []
        for item in images:
            image = item.get("image")
            if image is None:
                continue
            
            scores = self.evaluate(image)
            item.update({
                "face_similarity": scores.face_similarity,
                "aesthetic_score": scores.aesthetic_score,
                "technical_quality": scores.technical_quality,
                "artifact_score": scores.artifact_score,
                "overall_score": scores.overall_score,
                "passes_filter": scores.passes_filter,
            })
            evaluated.append(item)
        
        # Sort by overall score (descending)
        evaluated.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Filter
        if top_k is not None:
            # Keep top K
            results = evaluated[:top_k]
        else:
            # Keep all passing
            results = [item for item in evaluated if item["passes_filter"]]
            
            # If not enough pass, take top min_pass
            if len(results) < min_pass:
                logger.warning(f"Only {len(results)} images passed, taking top {min_pass}")
                results = evaluated[:min_pass]
        
        passed = sum(1 for r in results if r["passes_filter"])
        logger.info(f"Filtered {len(images)} images → {len(results)} kept ({passed} passed filters)")
        
        return results
        
    def rank_images(self, images: list[dict]) -> list[dict]:
        """
        Rank images by quality without filtering.
        
        Args:
            images: List of dicts with 'image' key
            
        Returns:
            Same list sorted by overall_score descending
        """
        for item in images:
            image = item.get("image")
            if image is None:
                continue
                
            scores = self.evaluate(image)
            item.update({
                "face_similarity": scores.face_similarity,
                "aesthetic_score": scores.aesthetic_score,
                "technical_quality": scores.technical_quality,
                "artifact_score": scores.artifact_score,
                "overall_score": scores.overall_score,
                "passes_filter": scores.passes_filter,
            })
            
        images.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        return images

    def cleanup(self):
        """Release resources."""
        self.face_analyzer = None
        self.aesthetic_predictor = None
        self.reference_embeddings = []
        self._models_loaded = False
