"""
Storage Service

S3-compatible object storage client for MinIO.
"""

import io
import logging
from typing import BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StorageService:
    """
    S3-compatible storage service.
    
    Handles file uploads, downloads, and presigned URL generation
    for MinIO object storage.
    """

    def __init__(self):
        """Initialize S3 client."""
        endpoint_url = f"{'https' if settings.minio_secure else 'http'}://{settings.minio_endpoint}"
        
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",  # Required for MinIO
        )
        self.bucket = settings.minio_bucket
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"Bucket '{self.bucket}' exists")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                logger.info(f"Creating bucket '{self.bucket}'")
                self.client.create_bucket(Bucket=self.bucket)
            else:
                logger.error(f"Error checking bucket: {e}")
                raise

    def upload_file(
        self,
        file: BinaryIO,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload a file to S3.
        
        Args:
            file: File-like object to upload
            key: S3 object key (path)
            content_type: MIME type of the file
            
        Returns:
            The S3 key of the uploaded file
        """
        try:
            self.client.upload_fileobj(
                file,
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info(f"Uploaded file to s3://{self.bucket}/{key}")
            return key
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            raise

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload bytes to S3.
        
        Args:
            data: Bytes to upload
            key: S3 object key (path)
            content_type: MIME type
            
        Returns:
            The S3 key of the uploaded file
        """
        return self.upload_file(io.BytesIO(data), key, content_type)

    def download_file(self, key: str) -> bytes:
        """
        Download a file from S3.
        
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

    def delete_file(self, key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Deleted s3://{self.bucket}/{key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    def get_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "get_object",
    ) -> str:
        """
        Generate a presigned URL for temporary access.
        
        Args:
            key: S3 object key
            expires_in: URL expiration in seconds (default 1 hour)
            method: S3 method (get_object, put_object)
            
        Returns:
            Presigned URL string
        """
        try:
            url = self.client.generate_presigned_url(
                method,
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    def list_files(self, prefix: str = "") -> list[dict]:
        """
        List files with a given prefix.
        
        Args:
            prefix: S3 key prefix to filter by
            
        Returns:
            List of file metadata dicts
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
            )
            return response.get("Contents", [])
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def file_exists(self, key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if file exists
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


# Singleton instance
_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    """Get or create storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
