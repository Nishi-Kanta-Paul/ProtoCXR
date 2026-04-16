"""Common utility helpers for ProtoCXR."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for deterministic behavior.

    Args:
        seed: Random seed value.

    Returns:
        None.

    Raises:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Track running average of scalar values.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self) -> None:
        """Initialize a fresh meter state.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self.reset()

    def reset(self) -> None:
        """Reset tracked statistics.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update running stats with a new observation.

        Args:
            val: Scalar value to add.
            n: Number of items represented by val.

        Returns:
            None.

        Raises:
            None.
        """

        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Data dictionary to save.
        path: Output path.

    Returns:
        None.

    Raises:
        OSError: If writing fails.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file if present.

    Args:
        path: JSON path to read.

    Returns:
        Parsed dict, or {} when file is missing.

    Raises:
        json.JSONDecodeError: If file contents are invalid JSON.
    """

    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def append_jsonl(data: Dict[str, Any], path: str) -> None:
    """Append one record to a JSONL file.

    Args:
        data: Dictionary record to append.
        path: JSONL destination path.

    Returns:
        None.

    Raises:
        OSError: If writing fails.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(data) + "\n")


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Create a logger with file and stream handlers.

    Args:
        name: Logger name.
        log_file: File path for logging output.

    Returns:
        Configured logger instance.

    Raises:
        OSError: If the log file cannot be created.
    """

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def make_dirs(config: Any) -> None:
    """Create project directories required for runs and outputs.

    Args:
        config: Config object with path attributes.

    Returns:
        None.

    Raises:
        OSError: If directory creation fails.
    """

    dirs = [
        config.DATA_DIR,
        config.EXPERIMENT_DIR,
        os.path.join(config.DRIVE_ROOT, "experiments", "densenet121"),
        os.path.join(config.DRIVE_ROOT, "experiments", "protopnet"),
        os.path.join(config.DRIVE_ROOT, "experiments", "cbm"),
        config.OUTPUT_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        os.path.join(config.EXPERIMENT_DIR, "checkpoints"),
        os.path.join(config.EXPERIMENT_DIR, "logs"),
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def get_device() -> torch.device:
    """Return current compute device.

    Args:
        None.

    Returns:
        torch.device("cuda") if available, else torch.device("cpu").

    Raises:
        None.
    """

    if torch.cuda.is_available():
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("Using device: cpu")
    return torch.device("cpu")


def mount_google_drive() -> None:
    """Mount Google Drive when running inside Colab.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    try:
        from google.colab import drive

        drive.mount("/content/drive")
        print("Drive mounted")
    except ImportError:
        print("Not in Colab, skipping mount.")