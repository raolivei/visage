"""
Storage Client

S3-compatible storage client for MinIO.
Same as API storage service but for worker use.
"""

import io
import logging
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StorageClient:
    """S3-compatible storage client."""

    def __init__(self):
        """Initialize S3 client."""
        endpoint_url = f"{'https' if settings.minio_secure else 'http'}://{settings.minio_endpoint}"
        
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.bucket = settings.minio_bucket

    def download_file(self, key: str, local_path: Path) -> Path:
        """
        Download a file from S3 to local filesystem.
        
        Args:
            key: S3 object key
            local_path: Local path to save file
            
        Returns:
            Path to downloaded file
        """
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, key, str(local_path))
            logger.info(f"Downloaded s3://{self.bucket}/{key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download file: {e}")
            raise

    def download_bytes(self, key: str) -> bytes:
        """
        Download a file from S3 as bytes.
        
        Args:
            key: S3 object key
            
        Returns:
            File contents as bytes
        """
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Failed to download file: {e}")
            raise

    def upload_file(self, local_path: Path, key: str, content_type: str = "image/png") -> str:
        """
        Upload a local file to S3.
        
        Args:
            local_path: Local file path
            key: S3 object key
            content_type: MIME type
            
        Returns:
            S3 key
        """
        try:
            self.client.upload_file(
                str(local_path),
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{key}")
            return key
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            raise

    def upload_bytes(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        """
        Upload bytes to S3.
        
        Args:
            data: Bytes to upload
            key: S3 object key
            content_type: MIME type
            
        Returns:
            S3 key
        """
        try:
            self.client.upload_fileobj(
                io.BytesIO(data),
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info(f"Uploaded bytes to s3://{self.bucket}/{key}")
            return key
        except ClientError as e:
            logger.error(f"Failed to upload bytes: {e}")
            raise

    def list_files(self, prefix: str) -> list[str]:
        """
        List files with a given prefix.
        
        Args:
            prefix: S3 key prefix
            
        Returns:
            List of S3 keys
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
            )
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []


# Singleton
_storage_client: StorageClient | None = None


def get_storage_client() -> StorageClient:
    """Get or create storage client singleton."""
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client
