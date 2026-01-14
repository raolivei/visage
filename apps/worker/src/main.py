"""
Visage GPU Worker

Main entry point for the background worker.
Polls for jobs and processes them.
"""

import io
import logging
import signal
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings, get_device
from .queue import get_queue_client
from .storage import get_storage_client
from .pipeline import LoRATrainer, ImageGenerator, QualityFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()

settings = get_settings()

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    logger.info("Shutdown signal received, finishing current job...")
    shutdown_requested = True


def update_pack_status(pack_id: str, status: str, error: str | None = None):
    """Update pack status via API."""
    try:
        # NOTE: In production, call the API to update pack status
        # httpx.patch(
        #     f"{settings.api_url}/api/packs/{pack_id}",
        #     json={"status": status, "error_message": error}
        # )
        logger.info(f"Pack {pack_id} status updated to {status}")
    except Exception as e:
        logger.error(f"Failed to update pack status: {e}")


def process_train_job(job: dict, queue, storage) -> dict:
    """
    Process a training job.
    
    1. Download user photos
    2. Train LoRA
    3. Upload LoRA weights
    4. Start generation jobs
    """
    job_id = job["id"]
    pack_id = job["pack_id"]
    params = job.get("parameters", {})
    
    logger.info(f"Processing training job {job_id} for pack {pack_id}")
    
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Progress callback
        def progress(pct: int, step: str):
            queue.update_progress(job_id, pct, step)
            queue.add_log(job_id, step)
        
        try:
            # Download photos
            progress(5, "Downloading photos")
            photo_keys = storage.list_files(f"packs/{pack_id}/photos/")
            
            if not photo_keys:
                raise ValueError("No photos found for pack")
            
            photo_paths = []
            for key in photo_keys:
                local_path = tmpdir / "photos" / Path(key).name
                storage.download_file(key, local_path)
                photo_paths.append(local_path)
            
            logger.info(f"Downloaded {len(photo_paths)} photos")
            
            # Train LoRA
            progress(10, "Starting LoRA training")
            trainer = LoRATrainer()
            
            lora_dir = tmpdir / "lora"
            lora_path = trainer.train(
                photo_paths=photo_paths,
                output_dir=lora_dir,
                trigger_token=f"@visage_{pack_id[:8]}",
                progress_callback=lambda p, s: progress(10 + int(p * 0.7), s),
            )
            
            # Upload LoRA weights
            progress(85, "Uploading LoRA weights")
            lora_s3_key = f"packs/{pack_id}/lora/lora_weights.safetensors"
            storage.upload_file(lora_path, lora_s3_key)
            
            # Cleanup
            trainer.cleanup()
            
            progress(90, "Training complete, queuing generation")
            
            return {
                "lora_path": lora_s3_key,
                "photo_count": len(photo_paths),
                "style_presets": params.get("style_presets", ["corporate"]),
            }
            
        except Exception as e:
            logger.error(f"Training job failed: {e}")
            raise


def process_generate_job(job: dict, queue, storage) -> dict:
    """
    Process a generation job.
    
    1. Load LoRA
    2. Generate images for each style
    3. Filter results
    4. Upload outputs
    """
    job_id = job["id"]
    pack_id = job["pack_id"]
    params = job.get("parameters", {})
    
    logger.info(f"Processing generation job {job_id} for pack {pack_id}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        def progress(pct: int, step: str):
            queue.update_progress(job_id, pct, step)
            queue.add_log(job_id, step)
        
        try:
            # Download LoRA
            progress(5, "Loading LoRA weights")
            lora_key = f"packs/{pack_id}/lora/lora_weights.safetensors"
            lora_path = tmpdir / "lora" / "lora_weights.safetensors"
            
            try:
                storage.download_file(lora_key, lora_path)
            except Exception:
                logger.warning("No LoRA found, generating without fine-tuning")
                lora_path = None
            
            # Initialize generator
            progress(10, "Initializing generator")
            generator = ImageGenerator()
            
            if lora_path and lora_path.exists():
                generator.load_lora(lora_path)
            
            # Get prompts for each style
            progress(15, "Preparing prompts")
            
            # Import shared prompts
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "packages"))
            from shared.prompts import get_prompt_for_style
            
            style_presets = params.get("style_presets", ["corporate"])
            num_per_style = params.get("num_images_per_style", 20)
            
            trigger_token = f"@visage_{pack_id[:8]}"
            prompts = []
            for style in style_presets:
                try:
                    prompt, negative = get_prompt_for_style(style, trigger_token)
                    prompts.append((style, prompt, negative))
                except ValueError:
                    logger.warning(f"Unknown style: {style}")
            
            # Generate images
            progress(20, f"Generating {len(prompts) * num_per_style} images")
            
            def gen_progress(p: int, s: str):
                # Map generation progress to 20-70%
                progress(20 + int(p * 0.5), s)
            
            generated = generator.generate_batch(
                prompts=prompts,
                num_per_style=num_per_style,
                progress_callback=gen_progress,
            )
            
            logger.info(f"Generated {len(generated)} images")
            
            # Filter images
            progress(70, "Filtering images")
            
            # Load reference photos for face comparison
            photo_keys = storage.list_files(f"packs/{pack_id}/photos/")
            ref_paths = []
            for key in photo_keys[:5]:  # Use first 5 photos as reference
                ref_path = tmpdir / "refs" / Path(key).name
                storage.download_file(key, ref_path)
                ref_paths.append(ref_path)
            
            quality_filter = QualityFilter()
            quality_filter.set_reference_images(ref_paths)
            
            filtered = quality_filter.filter_batch(generated)
            
            logger.info(f"Filtered to {len(filtered)} images")
            
            # Upload results
            progress(85, f"Uploading {len(filtered)} outputs")
            
            uploaded = []
            for i, item in enumerate(filtered):
                img = item["image"]
                style = item["style_id"]
                seed = item["seed"]
                
                # Save to buffer
                buf = io.BytesIO()
                img.save(buf, format="PNG", quality=95)
                buf.seek(0)
                
                # Upload
                output_key = f"packs/{pack_id}/outputs/{style}_{seed}.png"
                storage.upload_bytes(buf.read(), output_key)
                
                uploaded.append({
                    "s3_key": output_key,
                    "style_preset": style,
                    "seed": seed,
                    "prompt": item["prompt"],
                    "score": item["overall_score"],
                    "face_similarity": item["face_similarity"],
                })
                
                if (i + 1) % 10 == 0:
                    progress(85 + int((i / len(filtered)) * 10), f"Uploaded {i + 1}/{len(filtered)}")
            
            # Cleanup
            generator.cleanup()
            quality_filter.cleanup()
            
            progress(100, "Generation complete")
            
            return {
                "generated_count": len(generated),
                "filtered_count": len(filtered),
                "outputs": uploaded,
            }
            
        except Exception as e:
            logger.error(f"Generation job failed: {e}")
            raise


def process_job(job: dict, queue, storage) -> dict | None:
    """
    Process a job based on its type.
    """
    job_type = job.get("type")
    
    if job_type == "train":
        return process_train_job(job, queue, storage)
    elif job_type == "generate":
        return process_generate_job(job, queue, storage)
    else:
        logger.warning(f"Unknown job type: {job_type}")
        return None


def main():
    """Main worker loop."""
    global shutdown_requested
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    console.print("[bold green]Visage GPU Worker[/bold green]")
    console.print(f"Worker ID: {settings.worker_id}")
    console.print(f"Device: {get_device()}")
    console.print(f"API: {settings.api_url}")
    console.print(f"Redis: {settings.redis_url}")
    console.print("")
    
    queue = get_queue_client()
    storage = get_storage_client()
    
    # Check connections
    if not queue.health_check():
        logger.error("Failed to connect to Redis")
        sys.exit(1)
    
    logger.info("Connected to Redis")
    logger.info("Starting job polling loop...")
    
    while not shutdown_requested:
        try:
            # Poll for jobs
            job = queue.dequeue()
            
            if job is None:
                # No jobs, wait and retry
                time.sleep(settings.poll_interval)
                continue
            
            job_id = job["id"]
            job_type = job.get("type")
            pack_id = job.get("pack_id")
            
            logger.info(f"Processing job {job_id} (type={job_type}, pack={pack_id})")
            
            try:
                result = process_job(job, queue, storage)
                queue.complete_job(job_id, result=result)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Job {job_id} failed: {error_msg}")
                queue.complete_job(job_id, error=error_msg)
                
                # Update pack status to failed
                if pack_id:
                    update_pack_status(pack_id, "failed", error_msg)
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(5)
    
    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    main()
