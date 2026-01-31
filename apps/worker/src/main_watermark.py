"""
Visage Watermark Worker

Lightweight worker that only processes watermark removal jobs.
Designed for CPU-only execution on Kubernetes (ARM64/x64).
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

from .config import get_settings
from .queue import get_queue_client
from .storage import get_storage_client
# Import WatermarkRemover directly to avoid loading GPU-dependent modules
from .pipeline.watermark_remover import WatermarkRemover

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
                        output_key = input_key.replace("/originals/", "/")
                    else:
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


def main():
    """Main worker loop - watermark jobs only."""
    global shutdown_requested
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    console.print("[bold green]Visage Watermark Worker (CPU)[/bold green]")
    console.print(f"Worker ID: {settings.worker_id}")
    console.print(f"Redis: {settings.redis_url}")
    console.print(f"MinIO: {settings.minio_endpoint}")
    console.print("")
    
    queue = get_queue_client()
    storage = get_storage_client()
    
    # Check connections
    if not queue.health_check():
        logger.error("Failed to connect to Redis")
        sys.exit(1)
    
    logger.info("Connected to Redis")
    logger.info("Starting watermark job polling loop...")
    
    while not shutdown_requested:
        try:
            # Only process watermark jobs
            watermark_job = queue.dequeue_watermark_job()
            
            if watermark_job:
                job_id = watermark_job["id"]
                logger.info(f"Processing watermark job {job_id}")
                try:
                    process_watermark_job(watermark_job, queue, storage)
                except Exception as e:
                    logger.error(f"Watermark job {job_id} failed: {e}")
            else:
                # No jobs, wait and retry
                time.sleep(settings.poll_interval)
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(5)
    
    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    main()
