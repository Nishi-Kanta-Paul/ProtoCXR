"""
src/config.py
=============
Single source of truth for all hyperparameters, paths, and training settings
for the ProtoCXR project.

All other modules import from this file. No values are hardcoded elsewhere.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    """ProtoCXR global configuration.

    All paths, hyperparameters, loss weights, and figure style settings
    are defined here. Import and instantiate ``Config()`` at the top of
    every module.

    Example:
        >>> from src.config import Config
        >>> config = Config()
        >>> config.NUM_CLASSES
        14
    """

    # ─── Paths ────────────────────────────────────────────────────────────────
    DRIVE_ROOT: str = "/content/drive/MyDrive/ProtoCXR"
    DATA_DIR: str = "/content/drive/MyDrive/ProtoCXR/data"
    CHEXPERT_DIR: str = "/content/drive/MyDrive/ProtoCXR/data/CheXpert-v1.0-small"
    CHEXPERT_CSV: str = "/content/drive/MyDrive/ProtoCXR/data/CheXpert-v1.0-small/train.csv"
    NIH_DIR: str = "/content/drive/MyDrive/ProtoCXR/data/NIH_ChestXray14/images"
    NIH_CSV: str = "/content/drive/MyDrive/ProtoCXR/data/NIH_ChestXray14/Data_Entry_2017.csv"
    EXPERIMENT_DIR: str = "/content/drive/MyDrive/ProtoCXR/experiments/protocxr"
    OUTPUT_DIR: str = "/content/drive/MyDrive/ProtoCXR/outputs"
    FIGURES_DIR: str = "/content/drive/MyDrive/ProtoCXR/outputs/figures"
    TABLES_DIR: str = "/content/drive/MyDrive/ProtoCXR/outputs/tables"

    # ─── Dataset ──────────────────────────────────────────────────────────────
    SUBSET_FRAC: float = 0.20       # Colab free-tier constraint
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    UNCERTAIN_POLICY: str = "zeros"  # CheXpert: -1 → 0
    SEEDS: List[int] = field(default_factory=lambda: [42, 123, 456])

    # ─── Model ────────────────────────────────────────────────────────────────
    NUM_CLASSES: int = 14
    NUM_PROTO: int = 10              # K prototypes per class
    FEAT_DIM: int = 512              # Prototype vector dimension
    BACKBONE: str = "densenet121"    # timm model name
    BACKBONE_PRETRAINED: bool = True

    # ─── Training — 4-phase schedule ──────────────────────────────────────────
    TOTAL_EPOCHS: int = 45
    WARMUP_EPOCHS: int = 10          # Phase 1: backbone frozen
    JOINT_EPOCHS: int = 30           # Phase 2: all layers
    PUSH_EVERY: int = 5              # Phase 3: prototype push interval
    FINETUNE_EPOCHS: int = 5         # Phase 4: FC only
    LR_BACKBONE: float = 1e-5
    LR_PROTO: float = 3e-4
    LR_FC: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0

    # ─── Loss weights ─────────────────────────────────────────────────────────
    LAMBDA_ARA: float = 0.01
    LAMBDA_PDR: float = 0.05
    LAMBDA_SEP: float = 0.08
    PDR_SIGMA: float = 1.5           # Margin for diversity regularizer
    SIM_EPSILON: float = 1e-4        # Numerical stability in similarity

    # ─── Labels ───────────────────────────────────────────────────────────────
    CHEXPERT_LABELS: List[str] = field(default_factory=lambda: [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ])

    NIH_LABELS: List[str] = field(default_factory=lambda: [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ])

    # ─── Augmentation ─────────────────────────────────────────────────────────
    ROTATION_DEGREES: int = 10
    BRIGHTNESS_JITTER: float = 0.2
    CONTRAST_JITTER: float = 0.2

    # ─── Figure style ─────────────────────────────────────────────────────────
    FIG_DPI: int = 180
    FIG_FONT: str = "serif"
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        "blue":   "#5B8DB8",
        "teal":   "#5DADA0",
        "purple": "#8B7EC8",
        "orange": "#E8A45A",
        "green":  "#7EB87E",
        "gray":   "#888888",
    })
