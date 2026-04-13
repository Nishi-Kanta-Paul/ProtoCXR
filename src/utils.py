"""
src/utils.py
============
Shared utility functions and classes used across the ProtoCXR project.

Includes: set_seed, AverageMeter, JSON helpers, logging setup,
          directory creation, and device detection.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python's ``random``, NumPy, PyTorch (CPU and CUDA),
    and enforces cuDNN determinism.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Track and compute the running average of a scalar value.

    Typical use case is tracking loss values over a training epoch.

    Attributes:
        val: Most recently observed value.
        avg: Running mean of all observed values.
        sum: Cumulative sum of all observed values.
        count: Total number of observations.

    Example:
        >>> meter = AverageMeter()
        >>> meter.update(1.0, n=2)
        >>> meter.update(2.0, n=2)
        >>> meter.avg
        1.5
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all tracked statistics to zero.

        Returns:
            None
        """
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update meter with a new observation.

        Args:
            val: New scalar value to track.
            n: Number of samples this value represents (e.g. batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def save_json(data: Dict[str, Any], path: str) -> None:
    """Serialize a dictionary to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        data: Dictionary to serialize.
        path: Destination file path (string or path-like).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary, or an empty dict if the file does not exist.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(data: Dict[str, Any], path: str) -> None:
    """Append a single JSON object as a new line to a ``.jsonl`` file.

    Creates the file (and parent directories) if they do not exist.

    Args:
        data: Dictionary to serialize as one JSON line.
        path: Destination ``.jsonl`` file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Create a logger that writes to both a file and stdout.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        log_file: Path to the log file. Parent dirs are created if needed.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def make_dirs(config: Any) -> None:
    """Create all output directories defined in the config.

    Args:
        config: A ``Config`` instance whose string attributes ending in
                ``_DIR`` are treated as directory paths to create.
    """
    dirs = [
        config.DRIVE_ROOT,
        config.DATA_DIR,
        config.EXPERIMENT_DIR,
        config.OUTPUT_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        os.path.join(config.EXPERIMENT_DIR, "checkpoints"),
        os.path.join(config.EXPERIMENT_DIR, "logs"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Returns:
        ``torch.device("cuda")`` if a CUDA GPU is available,
        otherwise ``torch.device("cpu")``.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda ({name})")
        return torch.device("cuda")
    print("Using device: cpu")
    return torch.device("cpu")
