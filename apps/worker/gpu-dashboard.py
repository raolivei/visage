#!/usr/bin/env python3
"""
Real-time GPU Dashboard for Visage Worker
Similar to btop but focused on GPU and training progress
"""
import time
import subprocess
import sys
import os
from datetime import datetime

# ANSI escape codes
CLEAR_LINE = "\033[K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
HOME = "\033[H"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

def get_worker_status():
    """Check if worker is running and get PID."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'python.*src.main'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pid = result.stdout.strip().split('\n')[0]
            return True, pid
        return False, None
    except:
        return False, None

def get_job_progress():
    """Get current job progress from Redis."""
    try:
        result = subprocess.run(
            ['docker', 'exec', 'visage-redis', 'redis-cli', 
             'HGETALL', 'visage:jobs:428c53e5-93d5-44a0-9a9d-768a0c87655f:data'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            data = {}
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    key = lines[i].strip()
                    value = lines[i + 1].strip()
                    data[key] = value
            return {
                'progress': data.get('progress', '0'),
                'step': data.get('current_step', 'Waiting...'),
                'status': data.get('status', 'unknown'),
                'job_type': data.get('job_type', 'unknown')
            }
    except:
        pass
    return {'progress': '0', 'step': 'Waiting...', 'status': 'unknown', 'job_type': 'unknown'}

def format_progress_bar(pct, width=40):
    """Create a progress bar."""
    try:
        pct_int = int(float(pct))
        filled = int(width * pct_int / 100)
        return "█" * filled + "░" * (width - filled)
    except:
        return "░" * width

def get_status_color(status):
    """Get color for status."""
    status = status.lower()
    if status in ('running', 'processing', 'completed'):
        return GREEN
    elif status in ('pending', 'queued'):
        return YELLOW
    elif status in ('failed', 'error', 'stopped'):
        return RED
    return RESET

def print_line(text):
    """Print a line and clear to end."""
    print(f"{text}{CLEAR_LINE}")

def main():
    """Main dashboard loop."""
    # Clear screen once at start and hide cursor
    print("\033[2J", end="")
    print(HIDE_CURSOR, end="")
    
    try:
        while True:
            # Move cursor to home position (no clear = no flicker)
            print(HOME, end="")
            
            now = datetime.now().strftime('%H:%M:%S')
            
            # Header
            print_line(f"{BOLD}╔{'═' * 58}╗{RESET}")
            print_line(f"{BOLD}║{' ' * 15}VISAGE GPU DASHBOARD{' ' * 23}║{RESET}")
            print_line(f"{BOLD}╚{'═' * 58}╝{RESET}")
            print_line("")
            
            # Time
            print_line(f"  {DIM}Last update:{RESET} {now}")
            print_line("")
            
            # Worker Status
            worker_running, pid = get_worker_status()
            if worker_running:
                status_str = f"{GREEN}● RUNNING{RESET} (PID: {pid})"
            else:
                status_str = f"{RED}○ STOPPED{RESET}"
            print_line(f"  {BOLD}Worker:{RESET}     {status_str}")
            print_line("")
            
            # Job Info
            job = get_job_progress()
            job_type = job['job_type'].upper() if job['job_type'] != 'unknown' else '-'
            status = job['status'].upper() if job['status'] != 'unknown' else '-'
            status_color = get_status_color(job['status'])
            
            print_line(f"  {BOLD}Job Type:{RESET}   {job_type}")
            print_line(f"  {BOLD}Status:{RESET}     {status_color}{status}{RESET}")
            print_line("")
            
            # Progress
            try:
                pct = int(float(job['progress']))
            except:
                pct = 0
            bar = format_progress_bar(pct)
            print_line(f"  {BOLD}Progress:{RESET}   [{CYAN}{bar}{RESET}] {pct:3d}%")
            print_line(f"  {BOLD}Step:{RESET}       {job['step']}")
            print_line("")
            
            # GPU Info (static, no need to query each time)
            print_line(f"  {BOLD}GPU:{RESET}        Apple M1 Pro")
            print_line(f"  {BOLD}Cores:{RESET}      16 GPU cores")
            print_line(f"  {BOLD}Device:{RESET}     MPS (Metal Performance Shaders)")
            print_line("")
            
            # Hint
            print_line(f"  {DIM}Press Ctrl+C to exit{RESET}")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        pass
    finally:
        # Show cursor and clean exit
        print(SHOW_CURSOR, end="")
        print("\033[2J\033[H", end="")
        print("Dashboard stopped.\n")

if __name__ == "__main__":
    main()
