#!/usr/bin/env python3
"""
Queue Watcher Daemon

Monitors Redis queues for pending jobs and auto-starts the worker.
Designed to run as a macOS LaunchAgent for background operation.

Usage:
    ./scripts/queue-watcher.py                     # Run in foreground (uses venv)
    ./scripts/queue-watcher.py --install-agent    # Install macOS LaunchAgent
    ./scripts/queue-watcher.py --uninstall-agent  # Remove LaunchAgent
    ./scripts/queue-watcher.py --status           # Show status
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORKER_DIR = Path(__file__).parent.parent

# If no redis package, try to use venv
try:
    import redis
except ImportError:
    # Try activating venv
    venv_site = WORKER_DIR / ".venv" / "lib"
    if venv_site.exists():
        # Find site-packages
        for sp in venv_site.glob("python*/site-packages"):
            sys.path.insert(0, str(sp))
    
    try:
        import redis
    except ImportError:
        print("Error: redis package not installed")
        print(f"Run: cd {WORKER_DIR} && source .venv/bin/activate && pip install redis")
        sys.exit(1)

# Add parent to path for config
sys.path.insert(0, str(WORKER_DIR / "src"))

# Configuration
POLL_INTERVAL = 5  # seconds between queue checks
WORKER_IDLE_TIMEOUT = 60  # seconds to keep worker alive after queue empty
# Redis URLs - try NodePort first (direct), fallback to localhost (port-forward)
REDIS_NODEPORT = os.environ.get("REDIS_URL", "redis://192.168.2.201:30379/0")
REDIS_LOCALHOST = "redis://localhost:6379/0"
REDIS_URL = REDIS_NODEPORT  # Will be set in main() after testing connectivity
# WORKER_DIR already defined at top
PID_FILE = WORKER_DIR / "worker.pid"
WATCHER_PID_FILE = WORKER_DIR / "queue-watcher.pid"
LOG_FILE = WORKER_DIR / "queue-watcher.log"

# Queue keys
JOBS_QUEUE = "visage:jobs:pending"
WATERMARK_QUEUE = "visage:watermark:pending"

# Feature flags - which job types to watch for
WATCH_REGULAR_JOBS = True   # Train/generate jobs (need GPU, Mac only)
WATCH_WATERMARK_JOBS = False  # Watermark jobs now handled by k8s cluster

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ]
)
logger = logging.getLogger(__name__)

shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    logger.info("Shutdown signal received")
    shutdown_requested = True


def get_redis_client():
    """Create Redis connection."""
    return redis.from_url(REDIS_URL, decode_responses=True)


def get_pending_job_count(r: redis.Redis) -> dict:
    """Get counts of pending jobs by queue type (respects feature flags)."""
    try:
        jobs_count = r.zcard(JOBS_QUEUE) if WATCH_REGULAR_JOBS else 0
        watermark_count = r.llen(WATERMARK_QUEUE) if WATCH_WATERMARK_JOBS else 0
        return {
            "jobs": jobs_count,
            "watermark": watermark_count,
            "total": jobs_count + watermark_count,
        }
    except redis.RedisError as e:
        logger.error(f"Redis error: {e}")
        return {"jobs": 0, "watermark": 0, "total": 0}


def is_worker_running() -> bool:
    """Check if worker process is running."""
    if not PID_FILE.exists():
        return False
    
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        return True
    except (ValueError, OSError, ProcessLookupError):
        # PID file exists but process is not running
        PID_FILE.unlink(missing_ok=True)
        return False


def start_worker():
    """Start the worker process."""
    if is_worker_running():
        logger.info("Worker already running")
        return True
    
    logger.info("Starting worker...")
    
    # Ensure we're using the venv python
    venv_python = WORKER_DIR / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    
    # Load environment from .env.production if it exists
    env = os.environ.copy()
    env_file = WORKER_DIR / ".env.production"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    
    # Override Redis URL with what we're using
    env["REDIS_URL"] = REDIS_URL
    
    # Keep Mac awake during processing
    caffeinate_proc = subprocess.Popen(
        ["caffeinate", "-dims"],
        start_new_session=True,
    )
    
    # Start worker log file
    log_file = WORKER_DIR / "worker.log"
    log_handle = open(log_file, "a")
    
    # Start the worker
    process = subprocess.Popen(
        [python_exe, "-m", "src.main"],
        cwd=str(WORKER_DIR),
        env=env,
        start_new_session=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    
    # Write PID file
    PID_FILE.write_text(str(process.pid))
    logger.info(f"Worker started with PID {process.pid}")
    logger.info(f"Worker log: {log_file}")
    
    # Store caffeinate PID for cleanup
    (WORKER_DIR / "caffeinate.pid").write_text(str(caffeinate_proc.pid))
    
    return True


def send_notification(title: str, message: str):
    """Send macOS notification."""
    try:
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "{title}"'
        ], capture_output=True, timeout=5)
    except Exception:
        pass  # Notifications are optional


def ensure_port_forward():
    """Ensure Redis port-forward is running, start if needed."""
    import socket
    
    # Test if localhost:6379 is accessible
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 6379))
        sock.close()
        if result == 0:
            return True
    except:
        pass
    
    # Start port-forward
    logger.info("Starting Redis port-forward...")
    try:
        subprocess.Popen(
            ["kubectl", "port-forward", "-n", "visage", "svc/visage-redis", "6379:6379"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)
        return True
    except Exception as e:
        logger.warning(f"Failed to start port-forward: {e}")
        return False


def main():
    """Main watcher loop."""
    global shutdown_requested, REDIS_URL
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Write watcher PID file
    WATCHER_PID_FILE.write_text(str(os.getpid()))
    
    logger.info("=" * 50)
    logger.info("Visage Queue Watcher starting")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Worker dir: {WORKER_DIR}")
    logger.info("=" * 50)
    
    # Try to connect to Redis (NodePort first, then localhost with port-forward)
    r = None
    for url in [REDIS_NODEPORT, REDIS_LOCALHOST]:
        try:
            REDIS_URL = url
            r = redis.from_url(url, decode_responses=True, socket_timeout=2)
            r.ping()
            logger.info(f"Connected to Redis at {url}")
            break
        except redis.RedisError:
            continue
    
    if r is None:
        # Try setting up port-forward
        logger.info("Direct Redis connection failed, trying port-forward...")
        if ensure_port_forward():
            try:
                REDIS_URL = REDIS_LOCALHOST
                r = redis.from_url(REDIS_LOCALHOST, decode_responses=True, socket_timeout=2)
                r.ping()
                logger.info(f"Connected to Redis via port-forward")
            except redis.RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                sys.exit(1)
        else:
            logger.error("Failed to connect to Redis (no port-forward)")
            sys.exit(1)
    
    last_job_seen = time.time()
    worker_started_for_batch = False
    
    while not shutdown_requested:
        try:
            counts = get_pending_job_count(r)
            total = counts["total"]
            
            if total > 0:
                last_job_seen = time.time()
                
                if not is_worker_running():
                    logger.info(f"Jobs pending: {counts['jobs']} regular, {counts['watermark']} watermark")
                    send_notification(
                        "Visage Worker",
                        f"Starting worker for {total} pending job(s)"
                    )
                    start_worker()
                    worker_started_for_batch = True
                elif not worker_started_for_batch:
                    # Worker is running but we didn't start it (was already running)
                    logger.debug(f"Worker running, {total} jobs pending")
            else:
                # No jobs pending
                idle_time = time.time() - last_job_seen
                
                if is_worker_running():
                    if idle_time > WORKER_IDLE_TIMEOUT:
                        logger.info(f"Queue empty for {WORKER_IDLE_TIMEOUT}s, worker will exit naturally")
                        worker_started_for_batch = False
                    else:
                        logger.debug(f"Queue empty, worker idle for {idle_time:.0f}s")
            
            time.sleep(POLL_INTERVAL)
            
        except redis.RedisError as e:
            logger.error(f"Redis error: {e}, reconnecting...")
            time.sleep(10)
            try:
                r = get_redis_client()
                r.ping()
            except:
                pass
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(10)
    
    # Cleanup
    WATCHER_PID_FILE.unlink(missing_ok=True)
    logger.info("Queue watcher stopped")


def install_launch_agent():
    """Install macOS LaunchAgent for auto-start."""
    agent_name = "com.visage.queue-watcher"
    plist_path = Path.home() / "Library/LaunchAgents" / f"{agent_name}.plist"
    
    # Ensure LaunchAgents directory exists
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get paths
    python_path = sys.executable
    script_path = Path(__file__).resolve()
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{agent_name}</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{WORKER_DIR}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>REDIS_URL</key>
        <string>{REDIS_URL}</string>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>{LOG_FILE}</string>
    
    <key>StandardErrorPath</key>
    <string>{LOG_FILE}</string>
    
    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>
"""
    
    plist_path.write_text(plist_content)
    print(f"Created LaunchAgent at: {plist_path}")
    
    # Load the agent
    result = subprocess.run(
        ["launchctl", "load", str(plist_path)],
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        print("LaunchAgent loaded successfully")
        print(f"The queue watcher will now run automatically on login")
        print(f"Logs: {LOG_FILE}")
    else:
        print(f"Warning: launchctl load returned: {result.stderr}")
        print("You may need to run: launchctl load ~/Library/LaunchAgents/com.visage.queue-watcher.plist")


def uninstall_launch_agent():
    """Remove macOS LaunchAgent."""
    agent_name = "com.visage.queue-watcher"
    plist_path = Path.home() / "Library/LaunchAgents" / f"{agent_name}.plist"
    
    if plist_path.exists():
        # Unload first
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True,
        )
        
        plist_path.unlink()
        print(f"Removed LaunchAgent: {plist_path}")
    else:
        print("LaunchAgent not found")
    
    # Stop any running watcher
    if WATCHER_PID_FILE.exists():
        try:
            pid = int(WATCHER_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped queue watcher (PID {pid})")
        except:
            pass
        WATCHER_PID_FILE.unlink(missing_ok=True)


def status():
    """Show status of queue watcher and worker."""
    print("=" * 50)
    print("Visage Queue Watcher Status")
    print("=" * 50)
    
    # Watcher status
    if WATCHER_PID_FILE.exists():
        try:
            pid = int(WATCHER_PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Queue Watcher: Running (PID {pid})")
        except:
            print("Queue Watcher: Not running (stale PID file)")
    else:
        print("Queue Watcher: Not running")
    
    # Worker status
    if is_worker_running():
        pid = int(PID_FILE.read_text().strip())
        print(f"Worker: Running (PID {pid})")
    else:
        print("Worker: Not running")
    
    # Queue status - try both URLs
    connected = False
    for url in [REDIS_NODEPORT, REDIS_LOCALHOST]:
        try:
            r = redis.from_url(url, decode_responses=True, socket_timeout=2)
            r.ping()
            counts = get_pending_job_count(r)
            print(f"Redis: {url}")
            print(f"Pending Jobs: {counts['jobs']}")
            print(f"Pending Watermark: {counts['watermark']}")
            connected = True
            break
        except Exception:
            continue
    
    if not connected:
        print("Redis: Unable to connect (try: kubectl port-forward -n visage svc/visage-redis 6379:6379)")
    
    # LaunchAgent status
    plist_path = Path.home() / "Library/LaunchAgents/com.visage.queue-watcher.plist"
    if plist_path.exists():
        print(f"LaunchAgent: Installed")
    else:
        print(f"LaunchAgent: Not installed")
    
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visage Queue Watcher")
    parser.add_argument("--install-agent", action="store_true", help="Install macOS LaunchAgent")
    parser.add_argument("--uninstall-agent", action="store_true", help="Remove macOS LaunchAgent")
    parser.add_argument("--status", action="store_true", help="Show status")
    
    args = parser.parse_args()
    
    if args.install_agent:
        install_launch_agent()
    elif args.uninstall_agent:
        uninstall_launch_agent()
    elif args.status:
        status()
    else:
        main()
