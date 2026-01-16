"""
LoRA Trainer

Fine-tune SDXL with LoRA on user photos using PEFT.
Supports checkpointing for crash recovery and resume.
"""

import logging
import json
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

import torch
from PIL import Image
from tqdm import tqdm

from ..config import get_settings, get_device, get_torch_dtype
from ..metrics import update_training_progress

logger = logging.getLogger(__name__)
settings = get_settings()

# Checkpoint interval (every N steps)
CHECKPOINT_INTERVAL = 100


@dataclass
class TrainingConfig:
    """Configuration for LoRA training."""
    # Resolution
    resolution: int = 1024
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1500
    
    # LoRA configuration
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = None  # Set in __post_init__
    
    # Optimization
    mixed_precision: str = "fp16"
    use_8bit_adam: bool = False
    gradient_checkpointing: bool = True
    
    # Regularization
    prior_loss_weight: float = 1.0
    
    # Checkpointing
    checkpoint_interval: int = CHECKPOINT_INTERVAL  # Save every N steps
    resume_from_checkpoint: bool = True  # Auto-resume if checkpoint exists
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default SDXL LoRA targets
            self.target_modules = [
                "to_k",
                "to_q", 
                "to_v",
                "to_out.0",
            ]


@dataclass
class TrainingCheckpoint:
    """Training checkpoint for crash recovery."""
    step: int
    loss: float
    optimizer_state: dict
    lora_state: dict
    trigger_token: str
    config: dict
    
    @classmethod
    def load(cls, checkpoint_path: Path) -> Optional["TrainingCheckpoint"]:
        """Load checkpoint from disk."""
        if not checkpoint_path.exists():
            return None
        try:
            data = torch.load(checkpoint_path, map_location="cpu")
            return cls(
                step=data["step"],
                loss=data["loss"],
                optimizer_state=data["optimizer_state"],
                lora_state=data["lora_state"],
                trigger_token=data["trigger_token"],
                config=data["config"],
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def save(self, checkpoint_path: Path):
        """Save checkpoint to disk."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.step,
            "loss": self.loss,
            "optimizer_state": self.optimizer_state,
            "lora_state": self.lora_state,
            "trigger_token": self.trigger_token,
            "config": self.config,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at step {self.step}")


class LoRATrainer:
    """
    LoRA trainer for SDXL using PEFT.
    
    Trains a lightweight LoRA adapter on user photos to capture
    their unique facial features and style.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer with optional config."""
        self.config = config or TrainingConfig()
        self.device = get_device()
        self.dtype = get_torch_dtype()
        self.pipe = None
        self.tokenizer = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.unet = None
        self.vae = None
        self.noise_scheduler = None
        
    def _load_models(self):
        """Load all required models for training."""
        if self.unet is not None:
            return
            
        logger.info(f"Loading SDXL models for training on {self.device}...")
        
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        
        model_id = settings.model_id
        
        # Load tokenizers
        logger.info("Loading tokenizers...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2"
        )
        
        # Load text encoders
        logger.info("Loading text encoders...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=self.dtype
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=self.dtype
        )
        
        # Load VAE
        logger.info("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            settings.vae_id, torch_dtype=self.dtype
        )
        
        # Load U-Net
        logger.info("Loading U-Net...")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=self.dtype
        )
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        # Move models to device
        self.text_encoder.to(self.device)
        self.text_encoder_2.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)
        
        # Freeze models (only train LoRA)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        logger.info("Models loaded successfully")
        
    def _setup_lora(self):
        """Configure LoRA adapter on U-Net."""
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        logger.info(f"LoRA configured: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        
    def _generate_caption(self, image_path: Path, trigger_token: str, index: int) -> str:
        """
        Generate training caption for an image.
        
        Uses professional photography terminology and varied descriptions
        to help the model learn diverse contexts while maintaining identity.
        """
        # Professional headshot captions with photography terminology
        variations = [
            # Studio/corporate styles
            f"a professional corporate headshot photograph of {trigger_token} person, "
            "Rembrandt lighting, shallow depth of field, 85mm lens, Canon 5D, "
            "neutral gray background, sharp focus on eyes, high resolution",
            
            f"a high-end executive portrait of {trigger_token} person, "
            "soft key light with fill, professional studio setup, "
            "clean white background, confident expression, detailed skin texture",
            
            f"editorial portrait photograph of {trigger_token} person, "
            "natural window light, f/2.8 aperture, bokeh background, "
            "authentic expression, magazine quality, 4K detail",
            
            # LinkedIn/business styles
            f"a LinkedIn professional headshot of {trigger_token} person, "
            "soft diffused lighting, warm color temperature, "
            "approachable expression, business casual, sharp details",
            
            f"modern business portrait of {trigger_token} person, "
            "ring light setup, clean contemporary background, "
            "friendly professional demeanor, high definition",
            
            # Creative/natural styles
            f"a natural light portrait of {trigger_token} person, "
            "golden hour lighting, outdoor setting, "
            "genuine smile, professional photography, DSLR quality",
            
            f"environmental portrait of {trigger_token} person, "
            "cinematic color grading, soft shadows, "
            "natural pose, detailed facial features, 8K resolution",
            
            # Technical/detailed styles
            f"studio portrait photograph of {trigger_token} person, "
            "three-point lighting setup, medium format quality, "
            "perfect skin detail, catchlights in eyes, neutral expression",
        ]
        return variations[index % len(variations)]
        
    def _prepare_dataset(
        self,
        photo_paths: list[Path],
        trigger_token: str,
        progress_callback: Optional[Callable] = None,
    ) -> list[dict]:
        """
        Prepare training dataset with images and captions.
        
        Args:
            photo_paths: Paths to training images
            trigger_token: Trigger token for this person
            progress_callback: Optional progress callback
            
        Returns:
            List of training samples
        """
        dataset = []
        
        for i, path in enumerate(photo_paths):
            try:
                # Load and resize image
                image = Image.open(path).convert("RGB")
                image = image.resize(
                    (self.config.resolution, self.config.resolution),
                    Image.Resampling.LANCZOS
                )
                
                # Generate caption
                caption = self._generate_caption(path, trigger_token, i)
                
                dataset.append({
                    "image": image,
                    "caption": caption,
                    "path": str(path),
                })
                
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                
        logger.info(f"Prepared {len(dataset)} training samples")
        return dataset
        
    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt for SDXL."""
        # Tokenize
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_2 = self.tokenizer_2(
            prompt,
            padding="max_length", 
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                tokens.input_ids.to(self.device),
                output_hidden_states=True,
            )
            pooled_prompt_embeds = self.text_encoder_2(
                tokens_2.input_ids.to(self.device),
                output_hidden_states=True,
            )
            
            # SDXL uses concatenated embeddings
            prompt_embeds = torch.cat([
                prompt_embeds.hidden_states[-2],
                pooled_prompt_embeds.hidden_states[-2]
            ], dim=-1)
            
            pooled_embeds = pooled_prompt_embeds.text_embeds
            
        return prompt_embeds, pooled_embeds
        
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to latent space."""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
        return latents

    def train(
        self,
        photo_paths: list[Path],
        output_dir: Path,
        trigger_token: str = "@visageUser",
        progress_callback: Callable[[int, str], None] | None = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Path:
        """
        Train LoRA on user photos with checkpoint support.
        
        Args:
            photo_paths: List of paths to training images
            output_dir: Directory to save trained LoRA
            trigger_token: Trigger token for the LoRA
            progress_callback: Optional callback for progress updates
            checkpoint_dir: Directory for checkpoints (defaults to output_dir/checkpoints)
            
        Returns:
            Path to trained LoRA weights
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        lora_path = output_dir / "lora_weights.safetensors"
        config_path = output_dir / "training_config.json"
        
        # Setup checkpoint directory
        checkpoint_dir = checkpoint_dir or (output_dir / "checkpoints")
        checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
        
        if progress_callback:
            progress_callback(2, "Loading models...")
            
        # Load models
        self._load_models()
        
        if progress_callback:
            progress_callback(8, "Setting up LoRA...")
            
        # Setup LoRA
        self._setup_lora()
        
        if progress_callback:
            progress_callback(10, "Preparing dataset...")
            
        # Prepare dataset
        dataset = self._prepare_dataset(photo_paths, trigger_token, progress_callback)
        
        if len(dataset) < 5:
            raise ValueError(f"Not enough valid training images: {len(dataset)} (need at least 5)")
            
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # Enable gradient checkpointing to save memory
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        
        # Check for existing checkpoint
        start_step = 0
        checkpoint = None
        if self.config.resume_from_checkpoint:
            checkpoint = TrainingCheckpoint.load(checkpoint_path)
            if checkpoint and checkpoint.trigger_token == trigger_token:
                start_step = checkpoint.step
                optimizer.load_state_dict(checkpoint.optimizer_state)
                # Load LoRA state
                self.unet.load_state_dict(checkpoint.lora_state, strict=False)
                logger.info(f"Resuming training from checkpoint at step {start_step}")
                if progress_callback:
                    progress_callback(15, f"Resuming from step {start_step}...")
            elif checkpoint:
                logger.info("Checkpoint found but for different trigger token, starting fresh")
            
        if progress_callback and start_step == 0:
            progress_callback(15, "Starting training...")
            
        # Training loop
        global_step = start_step
        num_steps = self.config.max_train_steps
        accumulation_steps = self.config.gradient_accumulation_steps
        
        self.unet.train()
        
        pbar = tqdm(total=num_steps, desc="Training LoRA")
        step_start_time = time.time()
        current_loss = 0.0
        
        while global_step < num_steps:
            for sample in dataset:
                if global_step >= num_steps:
                    break
                    
                # Get sample
                image = sample["image"]
                caption = sample["caption"]
                
                # Encode prompt
                prompt_embeds, pooled_embeds = self._encode_prompt(caption)
                
                # Encode image to latents
                latents = self._encode_image(image)
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Sample timestep
                timesteps = torch.randint(
                    0, 
                    self.noise_scheduler.config.num_train_timesteps,
                    (1,),
                    device=self.device
                )
                
                # Add noise to latents
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Prepare additional embeddings for SDXL
                add_time_ids = torch.tensor([
                    [self.config.resolution, self.config.resolution, 0, 0, 
                     self.config.resolution, self.config.resolution]
                ], device=self.device, dtype=self.dtype)
                
                added_cond_kwargs = {
                    "text_embeds": pooled_embeds,
                    "time_ids": add_time_ids,
                }
                
                # Forward pass
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(model_pred, noise)
                loss = loss / accumulation_steps
                current_loss = loss.item() * accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights
                if (global_step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=current_loss)
                
                # Calculate step time for metrics
                step_time = time.time() - step_start_time
                step_start_time = time.time()
                
                # Update Prometheus metrics every step
                update_training_progress(
                    step=global_step,
                    total_steps=num_steps,
                    loss=current_loss,
                    lr=self.config.learning_rate,
                    step_time=step_time,
                    epoch=global_step // len(dataset) if dataset else 0
                )
                
                # Progress callback
                if progress_callback and global_step % 50 == 0:
                    progress = 15 + int((global_step / num_steps) * 75)
                    progress_callback(progress, f"Training step {global_step}/{num_steps}")
                
                # Save checkpoint periodically
                if global_step > 0 and global_step % self.config.checkpoint_interval == 0:
                    try:
                        checkpoint = TrainingCheckpoint(
                            step=global_step,
                            loss=current_loss,
                            optimizer_state=optimizer.state_dict(),
                            lora_state=self.unet.state_dict(),
                            trigger_token=trigger_token,
                            config={
                                "resolution": self.config.resolution,
                                "lora_rank": self.config.lora_rank,
                                "lora_alpha": self.config.lora_alpha,
                                "learning_rate": self.config.learning_rate,
                            }
                        )
                        checkpoint.save(checkpoint_path)
                        if progress_callback:
                            progress_callback(
                                15 + int((global_step / num_steps) * 75),
                                f"Checkpoint saved at step {global_step}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint: {e}")
                    
        pbar.close()
        
        if progress_callback:
            progress_callback(92, "Saving LoRA weights...")
            
        # Save LoRA weights
        logger.info(f"Saving LoRA weights to {lora_path}")
        self.unet.save_pretrained(output_dir)
        
        # Also save in safetensors format for compatibility
        from peft import PeftModel
        self.unet.save_pretrained(output_dir, safe_serialization=True)
        
        # Save training config
        config_dict = {
            "trigger_token": trigger_token,
            "resolution": self.config.resolution,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "train_steps": global_step,
            "learning_rate": self.config.learning_rate,
            "num_images": len(dataset),
            "base_model": settings.model_id,
        }
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
            
        if progress_callback:
            progress_callback(100, "Training complete!")
        
        # Clean up checkpoint after successful completion
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Checkpoint cleaned up after successful training")
        except Exception as e:
            logger.warning(f"Failed to clean up checkpoint: {e}")
            
        logger.info(f"LoRA training complete. Weights saved to {output_dir}")
        return lora_path

    def cleanup(self):
        """Release GPU memory."""
        del self.pipe
        del self.text_encoder
        del self.text_encoder_2
        del self.vae
        del self.unet
        
        self.pipe = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.vae = None
        self.unet = None
            
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
            
        logger.info("Trainer resources released")
