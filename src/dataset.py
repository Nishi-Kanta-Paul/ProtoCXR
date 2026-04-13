"""
src/dataset.py
==============
Dataset classes, transforms, stratified subset sampling, and DataLoader
construction for CheXpert and NIH ChestX-ray14.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from src.config import Config

logger = logging.getLogger(__name__)


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(train: bool = True, image_size: int = 224,
                   config: Optional[Config] = None) -> transforms.Compose:
    """Build the image transformation pipeline.

    Args:
        train: If ``True``, applies data-augmentation transforms.
               If ``False``, applies inference-only transforms.
        image_size: Target spatial resolution (square).
        config: ``Config`` instance. Uses default ``Config()`` if ``None``.

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    if config is None:
        config = Config()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=config.BRIGHTNESS_JITTER,
                contrast=config.CONTRAST_JITTER,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ─── CheXpert Dataset ─────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """PyTorch Dataset for CheXpert v1.0-small (frontal images only).

    Handles the uncertain label policy (``-1`` → ``0`` for "zeros" policy)
    and filters lateral views.

    Args:
        csv_path: Path to the CheXpert ``train.csv`` or ``valid.csv``.
        img_root: Root directory that contains the raw image files.
        transform: Optional torchvision transform pipeline.
        label_names: List of 14 label column names.
        uncertain_policy: How to handle ``-1`` labels.
                          ``"zeros"`` → replace with ``0``.
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        transform: Optional[transforms.Compose] = None,
        label_names: Optional[List[str]] = None,
        uncertain_policy: str = "zeros",
    ) -> None:
        super().__init__()
        if label_names is None:
            label_names = Config().CHEXPERT_LABELS

        self.img_root = img_root
        self.transform = transform
        self.label_names = label_names
        self.uncertain_policy = uncertain_policy

        df = pd.read_csv(csv_path)

        # Keep only frontal-view images
        df = df[df["Path"].str.contains("frontal", case=False, na=False)].reset_index(drop=True)

        # Resolve uncertain labels
        if uncertain_policy == "zeros":
            df[label_names] = df[label_names].replace(-1, 0)

        # Fill remaining NaN with 0
        df[label_names] = df[label_names].fillna(0)

        self.df = df

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            Integer number of samples.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and transform a single sample.

        Args:
            idx: Index of the sample to load.

        Returns:
            Tuple of ``(image_tensor, label_tensor)`` where:
            - ``image_tensor`` has shape ``(3, H, W)`` and dtype ``float32``
            - ``label_tensor`` has shape ``(14,)`` and dtype ``float32``
        """
        row = self.df.iloc[idx]

        # Construct absolute image path
        img_path = os.path.join(self.img_root, row["Path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(row[self.label_names].values.astype(np.float32),
                             dtype=torch.float32)
        return image, label


# ─── NIH ChestX-ray14 Dataset ─────────────────────────────────────────────────

class NIHDataset(Dataset):
    """PyTorch Dataset for the NIH ChestX-ray14 dataset.

    Parses the pipe-separated ``Finding Labels`` column into a binary
    multi-hot label vector.

    Args:
        csv_path: Path to ``Data_Entry_2017.csv``.
        img_dir: Directory containing image files.
        transform: Optional torchvision transform pipeline.
        label_names: List of 14 finding label names (NIH order).
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[transforms.Compose] = None,
        label_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if label_names is None:
            label_names = Config().NIH_LABELS

        self.img_dir = img_dir
        self.transform = transform
        self.label_names = label_names

        df = pd.read_csv(csv_path)

        # Build binary label matrix from pipe-separated finding labels
        def _parse_labels(finding_str: str) -> List[float]:
            findings = [f.strip() for f in str(finding_str).split("|")]
            return [1.0 if lbl in findings else 0.0 for lbl in label_names]

        labels_matrix = np.array(
            df["Finding Labels"].apply(_parse_labels).tolist(),
            dtype=np.float32,
        )
        self.image_names: List[str] = df["Image Index"].tolist()
        self.labels: np.ndarray = labels_matrix

    def __len__(self) -> int:
        """Return the total number of samples.

        Returns:
            Integer number of samples.
        """
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and transform a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple ``(image_tensor, label_tensor)``:
            - ``image_tensor``: ``(3, H, W)`` float32 tensor
            - ``label_tensor``: ``(14,)`` float32 binary vector
        """
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ─── Stratified Subset ────────────────────────────────────────────────────────

def get_stratified_subset(dataset: Dataset, frac: float, seed: int) -> Subset:
    """Return a stratified subset of a dataset.

    Uses ``StratifiedShuffleSplit`` from scikit-learn. The stratification
    key is the argmax of each sample's multi-hot label vector (proxy for
    the dominant pathology).

    Args:
        dataset: Full PyTorch ``Dataset`` whose ``__getitem__`` returns
                 ``(image, label_tensor)``.
        frac: Fraction of dataset to retain (e.g. ``0.20``).
        seed: Random seed for reproducibility.

    Returns:
        A :class:`torch.utils.data.Subset` containing *frac* of the
        original samples with preserved class distribution.
    """
    total = len(dataset)  # type: ignore[arg-type]

    # Collect all labels to build stratification key
    all_labels = []
    for i in range(total):
        _, label = dataset[i]  # type: ignore[index]
        all_labels.append(label.numpy() if isinstance(label, torch.Tensor) else label)
    all_labels_array = np.array(all_labels)

    # Argmax as proxy stratify key for multi-label
    strat_keys = np.argmax(all_labels_array, axis=1)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=frac, random_state=seed
    )
    _, subset_indices = next(
        splitter.split(np.zeros(total), strat_keys)
    )

    logger.info(
        "Subset: %d / %d samples (%.0f%%)",
        len(subset_indices), total, frac * 100,
    )
    print(f"Subset: {len(subset_indices)} / {total} samples ({frac * 100:.0f}%)")

    return Subset(dataset, subset_indices.tolist())


# ─── DataLoader Builder ───────────────────────────────────────────────────────

def build_dataloaders(
    dataset_name: str,
    config: Config,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build stratified subset dataloaders for a given dataset.

    1. Instantiates the full dataset with augmented + non-augmented transforms.
    2. Applies ``get_stratified_subset`` (``SUBSET_FRAC = 0.20``).
    3. Splits the subset into train / val / test (70 / 15 / 15).

    Args:
        dataset_name: One of ``"chexpert"`` or ``"nih"``.
        config: ``Config`` instance containing all hyperparameters and paths.
        seed: Random seed forwarded to the subset sampler and generators.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader)`` — three
        :class:`torch.utils.data.DataLoader` instances.

    Raises:
        ValueError: If ``dataset_name`` is not ``"chexpert"`` or ``"nih"``.
    """
    train_tf = get_transforms(train=True,  image_size=config.IMAGE_SIZE, config=config)
    eval_tf  = get_transforms(train=False, image_size=config.IMAGE_SIZE, config=config)

    if dataset_name == "chexpert":
        full_train = CheXpertDataset(
            csv_path=config.CHEXPERT_CSV,
            img_root=config.CHEXPERT_DIR,
            transform=train_tf,
            label_names=config.CHEXPERT_LABELS,
            uncertain_policy=config.UNCERTAIN_POLICY,
        )
        full_eval = CheXpertDataset(
            csv_path=config.CHEXPERT_CSV,
            img_root=config.CHEXPERT_DIR,
            transform=eval_tf,
            label_names=config.CHEXPERT_LABELS,
            uncertain_policy=config.UNCERTAIN_POLICY,
        )
    elif dataset_name == "nih":
        full_train = NIHDataset(  # type: ignore[assignment]
            csv_path=config.NIH_CSV,
            img_dir=config.NIH_DIR,
            transform=train_tf,
            label_names=config.NIH_LABELS,
        )
        full_eval = NIHDataset(   # type: ignore[assignment]
            csv_path=config.NIH_CSV,
            img_dir=config.NIH_DIR,
            transform=eval_tf,
            label_names=config.NIH_LABELS,
        )
    else:
        raise ValueError(f"Unknown dataset_name: '{dataset_name}'. Use 'chexpert' or 'nih'.")

    # Stratified 20% subset (using training-augmented dataset for indices)
    subset = get_stratified_subset(full_eval, frac=config.SUBSET_FRAC, seed=seed)

    n_total  = len(subset)  # type: ignore[arg-type]
    n_train  = int(0.70 * n_total)
    n_val    = int(0.15 * n_total)
    n_test   = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        subset, [n_train, n_val, n_test], generator=generator
    )

    # Swap the eval dataset's indices for training set
    # (train_sub should use augmented transforms)
    # Re-wrap train indices with augmented dataset directly:
    train_indices = [subset.indices[i] for i in train_sub.indices]  # type: ignore[attr-defined]
    val_indices   = [subset.indices[i] for i in val_sub.indices]    # type: ignore[attr-defined]
    test_indices  = [subset.indices[i] for i in test_sub.indices]   # type: ignore[attr-defined]

    train_subset = Subset(full_train, train_indices)
    val_subset   = Subset(full_eval,  val_indices)
    test_subset  = Subset(full_eval,  test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
