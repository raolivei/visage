#!/usr/bin/env python3
"""
Migration script to register outputs that exist in MinIO but not in the database.

Run this after a generation job completes where the old worker code was used.

Usage:
    cd apps/worker
    python scripts/register_missing_outputs.py --pack-id <pack-id>
"""
import argparse
import httpx
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from storage import StorageClient
from config import settings


def get_outputs_from_minio(storage: StorageClient, pack_id: str) -> list[dict]:
    """List all outputs in MinIO for a pack."""
    prefix = f"packs/{pack_id}/outputs/"
    
    try:
        keys = storage.list_files(prefix)
    except Exception as e:
        print(f"Error listing files: {e}")
        return []
    
    outputs = []
    for key in keys:
        if not key.endswith('.png'):
            continue
        
        # Parse filename: {style}_{seed}.png
        filename = Path(key).stem
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            style, seed = parts
            outputs.append({
                "s3_key": key,
                "style_preset": style,
                "seed": float(seed) if seed.isdigit() else None,
                "is_filtered_out": False,
            })
        else:
            print(f"Warning: Could not parse filename: {filename}")
    
    return outputs


def get_outputs_from_database(api_url: str, pack_id: str) -> set[str]:
    """Get S3 keys of outputs already in database."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{api_url}/api/packs/{pack_id}/outputs?include_filtered=true")
            response.raise_for_status()
            data = response.json()
            # The API doesn't return s3_key directly, so we need to query differently
            # For now, just return count
            print(f"Database has {data['total']} outputs for this pack")
            return set()  # Can't get s3_keys from current API
    except Exception as e:
        print(f"Error querying database: {e}")
        return set()


def register_outputs(api_url: str, pack_id: str, outputs: list[dict], job_id: str = None) -> bool:
    """Register outputs in database via API."""
    if not outputs:
        return True
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{api_url}/api/packs/{pack_id}/outputs/batch",
                json={
                    "job_id": job_id,
                    "outputs": outputs,
                },
            )
            response.raise_for_status()
            result = response.json()
            print(f"✅ Registered {result['created_count']} outputs in database")
            return True
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error registering outputs: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Register missing outputs in database")
    parser.add_argument("--pack-id", required=True, help="Pack ID to process")
    parser.add_argument("--job-id", help="Job ID to associate with outputs")
    parser.add_argument("--api-url", default=settings.api_url, help="API URL")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually register, just show what would be done")
    args = parser.parse_args()
    
    print(f"Pack ID: {args.pack_id}")
    print(f"API URL: {args.api_url}")
    print()
    
    # Initialize storage client
    storage = StorageClient()
    
    # Get outputs from MinIO
    print("📦 Scanning MinIO for outputs...")
    minio_outputs = get_outputs_from_minio(storage, args.pack_id)
    print(f"   Found {len(minio_outputs)} outputs in MinIO")
    
    if not minio_outputs:
        print("No outputs found in MinIO. Generation may not be complete yet.")
        return
    
    # Show breakdown by style
    styles = {}
    for o in minio_outputs:
        style = o["style_preset"]
        styles[style] = styles.get(style, 0) + 1
    
    print("\n📊 Breakdown by style:")
    for style, count in sorted(styles.items()):
        print(f"   {style}: {count} images")
    
    # Check database
    print("\n🔍 Checking database...")
    get_outputs_from_database(args.api_url, args.pack_id)
    
    if args.dry_run:
        print(f"\n🔹 DRY RUN: Would register {len(minio_outputs)} outputs")
        return
    
    # Register in database
    print(f"\n📝 Registering {len(minio_outputs)} outputs in database...")
    if register_outputs(args.api_url, args.pack_id, minio_outputs, args.job_id):
        print("\n✅ Migration complete!")
    else:
        print("\n❌ Migration failed!")


if __name__ == "__main__":
    main()
