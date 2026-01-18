"""
Prometheus metrics for Visage worker.
Exposes training progress, GPU stats, and job metrics.

Supports two modes:
- Local HTTP server (for local development)
- Pushgateway (for centralized monitoring on k3s)
"""
from prometheus_client import (
    Counter, Gauge, Histogram, Info, 
    start_http_server, push_to_gateway,
    REGISTRY
)
import threading
import time
import logging

from .config import get_settings

logger = logging.getLogger(__name__)

# Info metrics
worker_info = Info('visage_worker', 'Worker information')

# Job metrics
jobs_total = Counter('visage_jobs_total', 'Total jobs processed', ['job_type', 'status'])
jobs_in_progress = Gauge('visage_jobs_in_progress', 'Currently processing jobs', ['job_type'])
job_duration_seconds = Histogram(
    'visage_job_duration_seconds', 
    'Job processing duration',
    ['job_type'],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400]  # 1m, 5m, 10m, 30m, 1h, 2h, 4h
)

# Training metrics
training_step = Gauge('visage_training_step', 'Current training step')
training_total_steps = Gauge('visage_training_total_steps', 'Total training steps')
training_loss = Gauge('visage_training_loss', 'Current training loss')
training_learning_rate = Gauge('visage_training_learning_rate', 'Current learning rate')
training_epoch = Gauge('visage_training_epoch', 'Current training epoch')
training_progress_percent = Gauge('visage_training_progress_percent', 'Training progress percentage')
training_eta_seconds = Gauge('visage_training_eta_seconds', 'Estimated time remaining')
training_step_duration = Histogram(
    'visage_training_step_duration_seconds',
    'Duration per training step',
    buckets=[30, 60, 120, 180, 240, 300, 600]
)

# Generation metrics
images_generated = Counter('visage_images_generated_total', 'Total images generated', ['style'])
images_filtered = Counter('visage_images_filtered_total', 'Images filtered out by quality', ['reason'])
generation_duration = Histogram(
    'visage_generation_duration_seconds',
    'Image generation duration',
    buckets=[5, 10, 20, 30, 60, 120]
)

# Generation progress metrics (new)
generation_progress_percent = Gauge('visage_generation_progress_percent', 'Generation job progress percentage')
generation_current_style = Info('visage_generation_current_style', 'Currently generating style')
generation_styles_total = Gauge('visage_generation_styles_total', 'Total styles to generate')
generation_styles_completed = Gauge('visage_generation_styles_completed', 'Styles completed')
generation_images_per_style = Gauge('visage_generation_images_per_style', 'Images per style')
generation_current_image = Gauge('visage_generation_current_image', 'Current image number in style')
generation_eta_seconds = Gauge('visage_generation_eta_seconds', 'Estimated time remaining for generation')

# Quality metrics
quality_score = Histogram(
    'visage_image_quality_score',
    'Image quality scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
face_similarity_score = Histogram(
    'visage_face_similarity_score',
    'Face similarity scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Resource metrics
gpu_memory_used_bytes = Gauge('visage_gpu_memory_used_bytes', 'GPU memory used')
gpu_memory_total_bytes = Gauge('visage_gpu_memory_total_bytes', 'Total GPU memory')
gpu_utilization_percent = Gauge('visage_gpu_utilization_percent', 'GPU utilization percentage')

# Queue metrics
queue_pending_jobs = Gauge('visage_queue_pending_jobs', 'Jobs waiting in queue', ['job_type'])
queue_processing_jobs = Gauge('visage_queue_processing_jobs', 'Jobs currently processing')


class MetricsPusher:
    """Push metrics to Prometheus Pushgateway periodically."""
    
    def __init__(self, pushgateway_url: str, job_name: str = "visage-worker", 
                 push_interval: float = 15.0):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.push_interval = push_interval
        self._pusher_thread = None
        self._running = False
    
    def start(self, worker_id: str, device: str):
        """Start the metrics pusher thread."""
        worker_info.info({
            'worker_id': worker_id,
            'device': device,
            'version': '0.1.0'
        })
        
        def push_loop():
            self._running = True
            while self._running:
                try:
                    push_to_gateway(
                        self.pushgateway_url, 
                        job=self.job_name,
                        grouping_key={'worker_id': worker_id},
                        registry=REGISTRY
                    )
                    logger.debug(f"Pushed metrics to {self.pushgateway_url}")
                except Exception as e:
                    logger.warning(f"Failed to push metrics: {e}")
                time.sleep(self.push_interval)
        
        self._pusher_thread = threading.Thread(target=push_loop, daemon=True)
        self._pusher_thread.start()
        logger.info(f"Metrics pusher started, pushing to {self.pushgateway_url} every {self.push_interval}s")
    
    def stop(self):
        """Stop the metrics pusher."""
        self._running = False
    
    def push_now(self, worker_id: str):
        """Push metrics immediately."""
        try:
            push_to_gateway(
                self.pushgateway_url, 
                job=self.job_name,
                grouping_key={'worker_id': worker_id},
                registry=REGISTRY
            )
        except Exception as e:
            logger.warning(f"Failed to push metrics: {e}")


class MetricsServer:
    """HTTP server for Prometheus metrics (local development)."""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self._server_thread = None
        self._running = False
    
    def start(self, worker_id: str, device: str):
        """Start the metrics HTTP server."""
        worker_info.info({
            'worker_id': worker_id,
            'device': device,
            'version': '0.1.0'
        })
        
        def run_server():
            start_http_server(self.port)
            self._running = True
            while self._running:
                time.sleep(1)
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"Metrics server started on port {self.port}")
    
    def stop(self):
        """Stop the metrics server."""
        self._running = False


# Global metrics instance
_metrics_handler = None
_worker_id = "unknown"

def start_metrics_server(port: int = 9090, worker_id: str = "unknown", device: str = "unknown"):
    """
    Start metrics collection.
    
    Uses Pushgateway if PUSHGATEWAY_URL is set, otherwise starts local HTTP server.
    """
    global _metrics_handler, _worker_id
    _worker_id = worker_id
    
    settings = get_settings()
    
    if settings.pushgateway_url:
        # Use Pushgateway for remote/production
        _metrics_handler = MetricsPusher(
            pushgateway_url=settings.pushgateway_url,
            push_interval=settings.metrics_push_interval
        )
        _metrics_handler.start(worker_id, device)
        print(f"Metrics: pushing to {settings.pushgateway_url}")
    else:
        # Use local HTTP server for development
        _metrics_handler = MetricsServer(port=port)
        _metrics_handler.start(worker_id, device)
        print(f"Metrics: http://localhost:{port}/metrics")

def push_metrics_now():
    """Push metrics immediately (for pushgateway mode)."""
    global _metrics_handler, _worker_id
    if isinstance(_metrics_handler, MetricsPusher):
        _metrics_handler.push_now(_worker_id)

def update_training_progress(step: int, total_steps: int, loss: float, lr: float = 0.0, 
                              step_time: float = 0.0, epoch: int = 0):
    """Update training metrics."""
    training_step.set(step)
    training_total_steps.set(total_steps)
    training_loss.set(loss)
    training_learning_rate.set(lr)
    training_epoch.set(epoch)
    
    progress = (step / total_steps * 100) if total_steps > 0 else 0
    training_progress_percent.set(progress)
    
    if step_time > 0:
        training_step_duration.observe(step_time)
        remaining_steps = total_steps - step
        eta = remaining_steps * step_time
        training_eta_seconds.set(eta)
    
    # Push immediately for real-time updates
    push_metrics_now()

def update_job_status(job_type: str, status: str, in_progress: bool = False):
    """Update job status metrics."""
    jobs_total.labels(job_type=job_type, status=status).inc()
    if in_progress:
        jobs_in_progress.labels(job_type=job_type).set(1)
    else:
        jobs_in_progress.labels(job_type=job_type).set(0)
    push_metrics_now()

def record_job_duration(job_type: str, duration: float):
    """Record job duration."""
    job_duration_seconds.labels(job_type=job_type).observe(duration)
    push_metrics_now()

def record_image_generated(style: str, quality: float = 0.0, similarity: float = 0.0):
    """Record image generation metrics."""
    images_generated.labels(style=style).inc()
    if quality > 0:
        quality_score.observe(quality)
    if similarity > 0:
        face_similarity_score.observe(similarity)


def update_generation_progress(
    progress_percent: float,
    current_style: str,
    styles_total: int,
    styles_completed: int,
    images_per_style: int,
    current_image: int,
    eta_seconds: float = 0.0,
):
    """Update generation progress metrics for Grafana dashboard."""
    generation_progress_percent.set(progress_percent)
    generation_current_style.info({'style': current_style})
    generation_styles_total.set(styles_total)
    generation_styles_completed.set(styles_completed)
    generation_images_per_style.set(images_per_style)
    generation_current_image.set(current_image)
    if eta_seconds > 0:
        generation_eta_seconds.set(eta_seconds)
    
    # Push immediately for real-time updates
    push_metrics_now()

def record_image_filtered(reason: str):
    """Record filtered image."""
    images_filtered.labels(reason=reason).inc()

def update_queue_metrics(pending: dict, processing: int):
    """Update queue metrics."""
    for job_type, count in pending.items():
        queue_pending_jobs.labels(job_type=job_type).set(count)
    queue_processing_jobs.set(processing)
