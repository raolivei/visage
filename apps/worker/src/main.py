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
from .pipeline import LoRATrainer, ImageGenerator, QualityFilter, WatermarkRemover
from .metrics import (
    start_metrics_server,
    update_job_status,
    record_job_duration,
    update_queue_metrics,
    update_generation_progress,
    record_image_generated,
    jobs_in_progress,
)

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
    import httpx
    
    try:
        payload = {"status": status}
        if error:
            payload["error_message"] = error
            
        with httpx.Client(timeout=10.0) as client:
            response = client.patch(
                f"{settings.api_url}/api/packs/{pack_id}",
                json=payload,
            )
            response.raise_for_status()
        logger.info(f"Pack {pack_id} status updated to {status}")
    except Exception as e:
        logger.warning(f"Failed to update pack status: {e}")


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
    
    api_url = settings.api_url
    
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Progress callback - syncs to both Redis AND PostgreSQL for real-time UI
        def progress(pct: int, step: str):
            queue.update_progress(job_id, pct, step)
            queue.add_log(job_id, step)
            # Sync to PostgreSQL so web UI shows accurate progress
            sync_job_status_to_db(api_url, pack_id, job_id, "processing", pct, step)
        
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
            
            # Verify LoRA weights exist before upload
            if not lora_path.exists():
                raise FileNotFoundError(
                    f"LoRA weights not found at {lora_path} after training. "
                    f"Directory contents: {list(lora_dir.iterdir()) if lora_dir.exists() else 'dir not found'}"
                )
            
            logger.info(f"LoRA training complete. Weights at {lora_path} ({lora_path.stat().st_size} bytes)")
            
            # Upload LoRA weights IMMEDIATELY after training
            # Do this before any cleanup to prevent race conditions
            progress(85, "Uploading LoRA weights")
            lora_s3_key = f"packs/{pack_id}/lora/lora_weights.safetensors"
            storage.upload_file(lora_path, lora_s3_key)
            
            # Verify upload succeeded
            logger.info(f"LoRA weights uploaded to {lora_s3_key}")
            
            # Cleanup
            trainer.cleanup()
            
            progress(90, "Training complete, queuing generation")
            
            # AUTO-START GENERATION after training completes
            # This ensures the pipeline is seamless without manual intervention
            style_presets = params.get("style_presets", [
                "corporate", "creative", "studio", "executive", "natural"
            ])
            num_per_style = params.get("num_images_per_style", 20)
            
            logger.info(f"Auto-starting generation job for {len(style_presets)} styles")
            
            generation_job_id = queue.enqueue_job(
                pack_id=pack_id,
                job_type="generate",
                parameters={
                    "style_presets": style_presets,
                    "num_images_per_style": num_per_style,
                }
            )
            
            logger.info(f"Generation job queued: {generation_job_id}")
            progress(95, f"Generation queued ({len(style_presets)} styles)")
            
            return {
                "lora_path": lora_s3_key,
                "photo_count": len(photo_paths),
                "style_presets": style_presets,
                "generation_job_id": generation_job_id,
            }
            
        except Exception as e:
            logger.error(f"Training job failed: {e}")
            raise


def register_outputs_in_database(api_url: str, pack_id: str, job_id: str, outputs: list[dict]) -> bool:
    """
    Register outputs in the database via API call.
    
    This enables incremental saving - outputs are registered as each style
    completes rather than waiting for all generation to finish.
    """
    import httpx
    
    if not outputs:
        return True
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{api_url}/api/packs/{pack_id}/outputs/batch",
                json={
                    "job_id": job_id,
                    "outputs": outputs,
                },
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Registered {result['created_count']} outputs in database")
            return True
    except Exception as e:
        logger.error(f"Failed to register outputs in database: {e}")
        return False


def sync_job_status_to_db(
    api_url: str, 
    pack_id: str, 
    job_id: str, 
    status: str,
    progress: int = None,
    current_step: str = None,
    error_message: str = None
) -> bool:
    """
    Sync job status from Redis to PostgreSQL via API call.
    
    This keeps the PostgreSQL database in sync with the worker's progress
    so the web UI and API can show accurate status.
    """
    import httpx
    
    payload = {"status": status}
    if progress is not None:
        payload["progress"] = progress
    if current_step is not None:
        payload["current_step"] = current_step
    if error_message is not None:
        payload["error_message"] = error_message
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.patch(
                f"{api_url}/api/packs/{pack_id}/jobs/{job_id}",
                json=payload,
            )
            response.raise_for_status()
            return True
    except Exception as e:
        logger.warning(f"Failed to sync job status to database: {e}")
        return False


def process_generate_job(job: dict, queue, storage) -> dict:
    """
    Process a generation job with INCREMENTAL SAVING.
    
    For each style:
    1. Generate images for that style
    2. Filter results
    3. Upload to MinIO
    4. Register in database via API
    
    This ensures partial results are saved even if the job fails mid-way.
    """
    job_id = job["id"]
    pack_id = job["pack_id"]
    params = job.get("parameters", {})
    
    logger.info(f"Processing generation job {job_id} for pack {pack_id}")
    
    api_url = settings.api_url
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        def progress(pct: int, step: str):
            """Update progress in both Redis AND PostgreSQL for real-time UI updates."""
            queue.update_progress(job_id, pct, step)
            queue.add_log(job_id, step)
            # Sync to PostgreSQL so web UI shows accurate progress
            sync_job_status_to_db(api_url, pack_id, job_id, "processing", pct, step)
        
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
            progress(12, "Preparing prompts")
            
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
            
            # Load reference photos for face comparison (do once, reuse for all styles)
            progress(14, "Loading reference photos")
            photo_keys = storage.list_files(f"packs/{pack_id}/photos/")
            ref_paths = []
            for key in photo_keys[:5]:  # Use first 5 photos as reference
                ref_path = tmpdir / "refs" / Path(key).name
                storage.download_file(key, ref_path)
                ref_paths.append(ref_path)
            
            quality_filter = QualityFilter()
            quality_filter.set_reference_images(ref_paths)
            
            # Process each style INCREMENTALLY
            total_generated = 0
            total_filtered = 0
            all_uploaded = []
            
            # Progress allocation: 15-95% for generation (80% total, split by styles)
            style_progress_share = 80 / len(prompts) if prompts else 80
            
            for style_idx, (style_id, prompt, negative_prompt) in enumerate(prompts):
                style_start_progress = 15 + int(style_idx * style_progress_share)
                style_end_progress = 15 + int((style_idx + 1) * style_progress_share)
                
                progress(style_start_progress, f"Generating {style_id} (style {style_idx + 1}/{len(prompts)})")
                
                # Update Grafana metrics for this style
                update_generation_progress(
                    progress_percent=style_start_progress,
                    current_style=style_id,
                    styles_total=len(prompts),
                    styles_completed=style_idx,
                    images_per_style=num_per_style,
                    current_image=0,
                    eta_seconds=(len(prompts) - style_idx) * num_per_style * 180,  # ~3 min per image
                )
                
                # Generate images for this style
                def style_gen_progress(p: int, s: str):
                    # Map 0-100 to style's progress range (first 70% of style's share)
                    gen_range = (style_end_progress - style_start_progress) * 0.7
                    pct = style_start_progress + int((p / 100) * gen_range)
                    progress(pct, f"{style_id}: {s}")
                
                style_images = generator.generate_batch(
                    prompts=[(style_id, prompt, negative_prompt)],
                    num_per_style=num_per_style,
                    progress_callback=style_gen_progress,
                )
                
                logger.info(f"Generated {len(style_images)} images for {style_id}")
                total_generated += len(style_images)
                
                # Record metrics for each generated image
                for img_data in style_images:
                    record_image_generated(style=style_id)
                
                # Filter images for this style
                filter_progress = style_start_progress + int((style_end_progress - style_start_progress) * 0.75)
                progress(filter_progress, f"Filtering {style_id}")
                
                style_filtered = quality_filter.filter_batch(style_images)
                logger.info(f"Filtered to {len(style_filtered)} images for {style_id}")
                total_filtered += len(style_filtered)
                
                # Upload to MinIO and register in database
                upload_progress = style_start_progress + int((style_end_progress - style_start_progress) * 0.85)
                progress(upload_progress, f"Saving {style_id} outputs")
                
                style_outputs = []
                for item in style_filtered:
                    img = item["image"]
                    seed = item["seed"]
                    
                    # Save to buffer
                    buf = io.BytesIO()
                    img.save(buf, format="PNG", quality=95)
                    buf.seek(0)
                    
                    # Upload to MinIO
                    output_key = f"packs/{pack_id}/outputs/{style_id}_{seed}.png"
                    storage.upload_bytes(buf.read(), output_key)
                    
                    style_outputs.append({
                        "s3_key": output_key,
                        "style_preset": style_id,
                        "seed": seed,
                        "prompt_used": item.get("prompt", prompt),
                        "negative_prompt": negative_prompt,
                        "score": item.get("overall_score"),
                        "face_similarity": item.get("face_similarity"),
                        "is_filtered_out": not item.get("passes_filter", True),
                    })
                
                # Register in database via API (INCREMENTAL SAVE!)
                api_url = settings.api_url
                if register_outputs_in_database(api_url, pack_id, job_id, style_outputs):
                    logger.info(f"✅ Saved {len(style_outputs)} outputs for {style_id} to database")
                else:
                    logger.warning(f"⚠️ Failed to save {style_id} outputs to database (will retry at end)")
                
                all_uploaded.extend(style_outputs)
                
                progress(style_end_progress, f"Completed {style_id}")
                
                # Update metrics after style completion
                update_generation_progress(
                    progress_percent=style_end_progress,
                    current_style=style_id,
                    styles_total=len(prompts),
                    styles_completed=style_idx + 1,
                    images_per_style=num_per_style,
                    current_image=num_per_style,
                    eta_seconds=(len(prompts) - style_idx - 1) * num_per_style * 180,
                )
            
            # Cleanup
            generator.cleanup()
            quality_filter.cleanup()
            
            progress(100, "Generation complete")
            
            return {
                "generated_count": total_generated,
                "filtered_count": total_filtered,
                "outputs": all_uploaded,
            }
            
        except Exception as e:
            logger.error(f"Generation job failed: {e}")
            raise


def process_watermark_job(job: dict, queue, storage) -> dict:
    """
    Process a watermark removal job.
    
    1. Download images from input keys
    2. Remove watermarks using LaMa inpainting
    3. Upload cleaned images
    4. Update job status with output keys
    """
    job_id = job["id"]
    input_keys = job.get("input_keys", [])
    pack_id = job.get("pack_id", "")
    
    logger.info(f"Processing watermark job {job_id} with {len(input_keys)} images")
    
    output_keys = []
    errors = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Initialize watermark remover
        remover = WatermarkRemover()
        
        try:
            total = len(input_keys)
            
            for i, input_key in enumerate(input_keys):
                try:
                    # Update progress
                    progress = int((i / total) * 100)
                    queue.update_watermark_job(job_id, progress=progress)
                    
                    # Download image
                    local_path = tmpdir / f"input_{i}{Path(input_key).suffix}"
                    storage.download_file(input_key, local_path)
                    
                    # Remove watermark
                    result = remover.remove(local_path)
                    
                    # Generate output key
                    if pack_id:
                        # For pack integration, replace the original
                        output_key = input_key.replace("/originals/", "/")
                    else:
                        # For standalone, save to output directory
                        output_key = input_key.replace("/input/", "/output/")
                    
                    # Upload cleaned image
                    buf = io.BytesIO()
                    result.cleaned_image.save(buf, format="PNG", quality=95)
                    buf.seek(0)
                    storage.upload_bytes(buf.read(), output_key)
                    
                    output_keys.append(output_key)
                    
                    logger.info(
                        f"Processed {i+1}/{total}: watermark_detected={result.watermark_detected}, "
                        f"confidence={result.confidence:.2f}"
                    )
                    
                except Exception as e:
                    error_msg = f"Failed to process {input_key}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Cleanup
            remover.cleanup()
            
            # Update job status
            queue.update_watermark_job(
                job_id,
                status="completed",
                progress=100,
                output_keys=output_keys,
                errors=errors,
            )
            
            logger.info(f"Watermark job {job_id} completed: {len(output_keys)} processed, {len(errors)} errors")
            
            return {
                "processed": len(output_keys),
                "errors": len(errors),
                "output_keys": output_keys,
            }
            
        except Exception as e:
            logger.error(f"Watermark job {job_id} failed: {e}")
            queue.update_watermark_job(
                job_id,
                status="failed",
                errors=errors + [str(e)],
            )
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
    
    # Start Prometheus metrics server
    start_metrics_server(
        port=9090,
        worker_id=settings.worker_id,
        device=get_device()
    )
    console.print(f"Metrics: http://localhost:9090/metrics")
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
            # First check for watermark jobs (higher priority, quick processing)
            watermark_job = queue.dequeue_watermark_job()
            if watermark_job:
                job_id = watermark_job["id"]
                logger.info(f"Processing watermark job {job_id}")
                try:
                    process_watermark_job(watermark_job, queue, storage)
                except Exception as e:
                    logger.error(f"Watermark job {job_id} failed: {e}")
                continue  # Check for more watermark jobs before other jobs
            
            # Poll for regular jobs
            job = queue.dequeue()
            
            if job is None:
                # Update queue metrics
                update_queue_metrics({"train": 0, "generate": 0}, 0)
                # No jobs, wait and retry
                time.sleep(settings.poll_interval)
                continue
            
            job_id = job["id"]
            job_type = job.get("type", "unknown")
            pack_id = job.get("pack_id")
            
            logger.info(f"Processing job {job_id} (type={job_type}, pack={pack_id})")
            
            # Update metrics
            update_job_status(job_type, "processing", in_progress=True)
            jobs_in_progress.labels(job_type=job_type).set(1)
            job_start_time = time.time()
            
            # Sync job status to PostgreSQL (started)
            sync_job_status_to_db(
                settings.api_url, pack_id, job_id, 
                "processing", progress=0, current_step="Starting..."
            )
            
            try:
                result = process_job(job, queue, storage)
                queue.complete_job(job_id, result=result)
                
                # Record success metrics
                job_duration = time.time() - job_start_time
                record_job_duration(job_type, job_duration)
                update_job_status(job_type, "completed", in_progress=False)
                
                # Sync job completion to PostgreSQL
                sync_job_status_to_db(
                    settings.api_url, pack_id, job_id,
                    "completed", progress=100, current_step="Complete"
                )
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Job {job_id} failed: {error_msg}")
                queue.complete_job(job_id, error=error_msg)
                
                # Record failure metrics
                update_job_status(job_type, "failed", in_progress=False)
                
                # Sync job failure to PostgreSQL
                sync_job_status_to_db(
                    settings.api_url, pack_id, job_id,
                    "failed", error_message=error_msg
                )
                
                # Update pack status to failed
                if pack_id:
                    update_pack_status(pack_id, "failed", error_msg)
            
            finally:
                jobs_in_progress.labels(job_type=job_type).set(0)
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(5)
    
    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    main()
