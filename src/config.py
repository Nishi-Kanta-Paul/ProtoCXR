"""Central project configuration for ProtoCXR."""

from dataclasses import dataclass, field
import os
from typing import Dict, List


@dataclass
class Config:
    """Stores every configurable value used across the project.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    # Paths
    DRIVE_ROOT: str = "/content/drive/MyDrive/ProtoCXR"
    DATA_DIR: str = "/content/drive/MyDrive/ProtoCXR/data/vindr-cxr"
    TRAIN_CSV: str = "/content/drive/MyDrive/ProtoCXR/data/vindr-cxr/train.csv"
    TEST_CSV: str = "/content/drive/MyDrive/ProtoCXR/data/vindr-cxr/test.csv"
    TRAIN_IMG_DIR: str = "/content/drive/MyDrive/ProtoCXR/data/vindr-cxr/train"
    TEST_IMG_DIR: str = "/content/drive/MyDrive/ProtoCXR/data/vindr-cxr/test"
    EXPERIMENT_DIR: str = "/content/drive/MyDrive/ProtoCXR/experiments/protocxr"
    OUTPUT_DIR: str = "/content/drive/MyDrive/ProtoCXR/outputs"
    FIGURES_DIR: str = "/content/drive/MyDrive/ProtoCXR/outputs/figures"
    TABLES_DIR: str = "/content/drive/MyDrive/ProtoCXR/outputs/tables"

    # Dataset
    LABELS: List[str] = field(default_factory=lambda: [
        "Aortic enlargement",
        "Cardiomegaly",
        "Pleural effusion",
        "Pleural thickening",
        "Pulmonary fibrosis",
        "No finding",
    ])
    NUM_CLASSES: int = 6
    VAL_SPLIT: float = 0.15
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    MAJORITY_VOTE_THRESHOLD: int = 2

    # Model
    NUM_PROTO: int = 10
    FEAT_DIM: int = 512
    BACKBONE: str = "densenet121"
    BACKBONE_PRETRAINED: bool = True

    # Training
    TOTAL_EPOCHS: int = 45
    WARMUP_EPOCHS: int = 10
    JOINT_EPOCHS: int = 30
    PUSH_EVERY: int = 5
    FINETUNE_EPOCHS: int = 5
    LR_BACKBONE: float = 1e-5
    LR_PROTO: float = 3e-4
    LR_FC: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    SEEDS: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Losses
    LAMBDA_ARA: float = 0.01
    LAMBDA_PDR: float = 0.05
    LAMBDA_SEP: float = 0.08
    PDR_SIGMA: float = 1.5
    SIM_EPSILON: float = 1e-4

    # Figure style
    FIG_DPI: int = 180
    FIG_FONT: str = "serif"
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        "blue": "#5B8DB8",
        "teal": "#5DADA0",
        "purple": "#8B7EC8",
        "orange": "#E8A45A",
        "green": "#7EB87E",
        "gray": "#888888",
    })

    def __post_init__(self) -> None:
        """Resolve runtime paths for Colab and local execution.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        default_drive_root = "/content/drive/MyDrive/ProtoCXR"
        env_drive_root = os.environ.get("PROTOCXR_DRIVE_ROOT")

        if env_drive_root:
            self.DRIVE_ROOT = env_drive_root
        elif self.DRIVE_ROOT == default_drive_root and not os.path.exists("/content/drive/MyDrive"):
            self.DRIVE_ROOT = os.path.abspath(os.path.join(os.getcwd(), "protocxr_runs"))

        self.DATA_DIR = os.path.join(self.DRIVE_ROOT, "data", "vindr-cxr")
        self.TRAIN_CSV = os.path.join(self.DATA_DIR, "train.csv")
        self.TEST_CSV = os.path.join(self.DATA_DIR, "test.csv")
        self.TRAIN_IMG_DIR = os.path.join(self.DATA_DIR, "train")
        self.TEST_IMG_DIR = os.path.join(self.DATA_DIR, "test")
        self.EXPERIMENT_DIR = os.path.join(self.DRIVE_ROOT, "experiments", "protocxr")
        self.OUTPUT_DIR = os.path.join(self.DRIVE_ROOT, "outputs")
        self.FIGURES_DIR = os.path.join(self.OUTPUT_DIR, "figures")
        self.TABLES_DIR = os.path.join(self.OUTPUT_DIR, "tables")
