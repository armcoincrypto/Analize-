"""
Hashing utilities for data integrity and reproducibility.
"""

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def compute_data_hash(data: pd.DataFrame | dict | list | str, length: int = 16) -> str:
    """
    Compute SHA256 hash of data.

    Args:
        data: Data to hash (DataFrame, dict, list, or string)
        length: Length of hash to return

    Returns:
        Hex hash string
    """
    if isinstance(data, pd.DataFrame):
        data_str = data.to_json(orient="records", date_format="iso")
    elif isinstance(data, (dict, list)):
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)

    return hashlib.sha256(data_str.encode()).hexdigest()[:length]


def compute_file_hash(file_path: Path | str, length: int = 16) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file
        length: Length of hash to return

    Returns:
        Hex hash string
    """
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()[:length]


def compute_config_hash(config: dict[str, Any], length: int = 16) -> str:
    """
    Compute hash of configuration for reproducibility tracking.

    Args:
        config: Configuration dictionary
        length: Length of hash to return

    Returns:
        Hex hash string
    """
    import json

    # Sort keys for deterministic hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:length]
