"""
S3-compatible object storage for raw data and exports.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import aioboto3
from botocore.config import Config

from analize.config import get_settings


class S3Storage:
    """Manages S3-compatible object storage."""

    def __init__(
        self,
        bucket: str | None = None,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ):
        settings = get_settings()
        self.bucket = bucket or settings.storage.s3_bucket
        self.endpoint = endpoint or settings.storage.s3_endpoint
        self.access_key = access_key or settings.storage.s3_access_key
        self.secret_key = secret_key or settings.storage.s3_secret_key
        self.region = region or settings.storage.s3_region

        self._session = aioboto3.Session()

    def _get_client_config(self) -> dict[str, Any]:
        """Get S3 client configuration."""
        config: dict[str, Any] = {
            "region_name": self.region,
        }

        if self.access_key and self.secret_key:
            config["aws_access_key_id"] = self.access_key
            config["aws_secret_access_key"] = self.secret_key

        if self.endpoint:
            config["endpoint_url"] = self.endpoint

        return config

    async def upload_file(
        self,
        local_path: Path | str,
        s3_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload a file to S3.

        Args:
            local_path: Local file path
            s3_key: S3 object key
            content_type: Optional content type
            metadata: Optional metadata

        Returns:
            S3 URI of uploaded file
        """
        extra_args: dict[str, Any] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        async with self._session.client("s3", **self._get_client_config()) as s3:
            await s3.upload_file(
                str(local_path),
                self.bucket,
                s3_key,
                ExtraArgs=extra_args if extra_args else None,
            )

        return f"s3://{self.bucket}/{s3_key}"

    async def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload bytes to S3.

        Args:
            data: Bytes to upload
            s3_key: S3 object key
            content_type: Optional content type
            metadata: Optional metadata

        Returns:
            S3 URI of uploaded object
        """
        extra_args: dict[str, Any] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        async with self._session.client("s3", **self._get_client_config()) as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=data,
                **extra_args,
            )

        return f"s3://{self.bucket}/{s3_key}"

    async def download_file(
        self,
        s3_key: str,
        local_path: Path | str,
    ) -> Path:
        """
        Download a file from S3.

        Args:
            s3_key: S3 object key
            local_path: Local destination path

        Returns:
            Local file path
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        async with self._session.client("s3", **self._get_client_config()) as s3:
            await s3.download_file(self.bucket, s3_key, str(local_path))

        return local_path

    async def download_bytes(self, s3_key: str) -> bytes:
        """
        Download bytes from S3.

        Args:
            s3_key: S3 object key

        Returns:
            Downloaded bytes
        """
        async with self._session.client("s3", **self._get_client_config()) as s3:
            response = await s3.get_object(Bucket=self.bucket, Key=s3_key)
            return await response["Body"].read()

    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        List objects in S3 with prefix.

        Args:
            prefix: S3 key prefix
            max_keys: Maximum number of keys to return

        Returns:
            List of object metadata
        """
        objects = []

        async with self._session.client("s3", **self._get_client_config()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix,
                PaginationConfig={"MaxItems": max_keys},
            ):
                for obj in page.get("Contents", []):
                    objects.append({
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                        "etag": obj["ETag"],
                    })

        return objects

    async def delete_object(self, s3_key: str) -> bool:
        """
        Delete an object from S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if deleted successfully
        """
        async with self._session.client("s3", **self._get_client_config()) as s3:
            await s3.delete_object(Bucket=self.bucket, Key=s3_key)
        return True

    async def delete_objects(self, s3_keys: list[str]) -> int:
        """
        Delete multiple objects from S3.

        Args:
            s3_keys: List of S3 object keys

        Returns:
            Number of objects deleted
        """
        if not s3_keys:
            return 0

        async with self._session.client("s3", **self._get_client_config()) as s3:
            response = await s3.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": key} for key in s3_keys]},
            )

        return len(response.get("Deleted", []))

    async def object_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if object exists
        """
        async with self._session.client("s3", **self._get_client_config()) as s3:
            try:
                await s3.head_object(Bucket=self.bucket, Key=s3_key)
                return True
            except Exception:
                return False

    async def get_presigned_url(
        self,
        s3_key: str,
        expires_in: int = 3600,
        method: str = "get_object",
    ) -> str:
        """
        Generate a presigned URL for S3 object.

        Args:
            s3_key: S3 object key
            expires_in: URL expiration in seconds
            method: S3 method (get_object or put_object)

        Returns:
            Presigned URL
        """
        async with self._session.client("s3", **self._get_client_config()) as s3:
            url = await s3.generate_presigned_url(
                method,
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expires_in,
            )
        return url

    async def ensure_bucket_exists(self) -> bool:
        """
        Ensure the S3 bucket exists, create if not.

        Returns:
            True if bucket exists or was created
        """
        async with self._session.client("s3", **self._get_client_config()) as s3:
            try:
                await s3.head_bucket(Bucket=self.bucket)
                return True
            except Exception:
                try:
                    await s3.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.region}
                        if self.region != "us-east-1"
                        else {},
                    )
                    return True
                except Exception:
                    return False

    async def get_bucket_stats(self) -> dict[str, Any]:
        """
        Get bucket statistics.

        Returns:
            Bucket statistics
        """
        total_size = 0
        total_objects = 0

        async with self._session.client("s3", **self._get_client_config()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.bucket):
                for obj in page.get("Contents", []):
                    total_size += obj["Size"]
                    total_objects += 1

        return {
            "bucket": self.bucket,
            "total_objects": total_objects,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
        }
