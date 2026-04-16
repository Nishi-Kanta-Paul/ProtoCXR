"""Data loading utilities for VinDr-CXR."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.config import Config


class VinDrCXRDataset(Dataset):
    """VinDr-CXR dataset with majority-vote label construction.

    Args:
        csv_path: Path to train.csv or test.csv.
        img_dir: Directory containing PNG and/or DICOM images.
        labels: List of diagnosis labels used for training.
        transform: Optional image transform pipeline.
        split: Dataset split name, either "train" or "test".
        vote_thresh: Minimum number of radiologists that must agree.

    Returns:
        None.

    Raises:
        ValueError: If split is not "train" or "test".
        FileNotFoundError: If input csv_path is missing.
    """

    _label_cache: Dict[Tuple[str, str, int, Tuple[str, ...]], pd.DataFrame] = {}

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        labels: List[str],
        transform: Optional[transforms.Compose],
        split: str,
        vote_thresh: int,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.csv_path = csv_path
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.split = split
        self.vote_thresh = vote_thresh

        self.label_df = self._build_label_df()
        self.image_ids = self.label_df["image_id"].tolist()

    def _build_label_df(self) -> pd.DataFrame:
        """Builds a one-row-per-image binary label table.

        Args:
            None.

        Returns:
            A dataframe with columns ["image_id"] + labels.

        Raises:
            ValueError: If required csv columns are missing.
        """

        cache_key = (
            self.csv_path,
            self.split,
            self.vote_thresh,
            tuple(self.labels),
        )
        if cache_key in VinDrCXRDataset._label_cache:
            return VinDrCXRDataset._label_cache[cache_key].copy()

        df = pd.read_csv(self.csv_path)
        required_cols = {"image_id", "class_name"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {sorted(missing)}"
            )

        image_ids = pd.DataFrame({"image_id": sorted(df["image_id"].astype(str).unique())})

        if self.split == "train":
            if "rad_id" not in df.columns:
                raise ValueError(
                    "Train CSV must include a 'rad_id' column for majority voting."
                )

            present_df = df[df["class_name"].isin(self.labels)].copy()
            present_df = present_df[["image_id", "rad_id", "class_name"]]
            present_df = present_df.drop_duplicates()
            vote_counts = (
                present_df.groupby(["image_id", "class_name"])["rad_id"]
                .nunique()
                .unstack(fill_value=0)
            )

            label_df = image_ids.copy()
            for label in self.labels:
                counts = vote_counts[label] if label in vote_counts.columns else 0
                if isinstance(counts, int):
                    label_df[label] = 0.0
                else:
                    label_df[label] = (
                        label_df["image_id"].map(counts).fillna(0) >= self.vote_thresh
                    ).astype(np.float32)
        else:
            if set(self.labels).issubset(df.columns):
                label_df = df[["image_id"] + self.labels].copy()
                for label in self.labels:
                    label_df[label] = (label_df[label].fillna(0) > 0).astype(np.float32)
            else:
                present_df = df[df["class_name"].isin(self.labels)][["image_id", "class_name"]]
                present_df = present_df.drop_duplicates()
                consensus = (
                    present_df.assign(value=1.0)
                    .pivot(index="image_id", columns="class_name", values="value")
                    .fillna(0.0)
                )
                label_df = image_ids.copy()
                for label in self.labels:
                    values = consensus[label] if label in consensus.columns else 0
                    if isinstance(values, int):
                        label_df[label] = 0.0
                    else:
                        label_df[label] = label_df["image_id"].map(values).fillna(0.0)

        label_df["image_id"] = label_df["image_id"].astype(str)
        for label in self.labels:
            label_df[label] = label_df[label].astype(np.float32)

        VinDrCXRDataset._label_cache[cache_key] = label_df.copy()
        return label_df

    def _resolve_image_path(self, image_id: str) -> str:
        """Resolve image path for an image id.

        Args:
            image_id: VinDr image id without extension.

        Returns:
            The absolute path to an existing image file.

        Raises:
            FileNotFoundError: If no image file is found.
        """

        candidates = [
            os.path.join(self.img_dir, image_id),
            os.path.join(self.img_dir, f"{image_id}.png"),
            os.path.join(self.img_dir, f"{image_id}.jpg"),
            os.path.join(self.img_dir, f"{image_id}.jpeg"),
            os.path.join(self.img_dir, f"{image_id}.dcm"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No image found for image_id={image_id} in {self.img_dir}")

    def _load_image(self, image_id: str) -> Image.Image:
        """Load PNG/JPG/DICOM image and convert to RGB PIL format.

        Args:
            image_id: VinDr image id without extension.

        Returns:
            A PIL RGB image.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """

        path = self._resolve_image_path(image_id)
        ext = os.path.splitext(path)[1].lower()

        if ext in {".png", ".jpg", ".jpeg"}:
            return Image.open(path).convert("RGB")

        if ext == ".dcm" or ext == "":
            dcm = pydicom.dcmread(path)
            array = dcm.pixel_array.astype(np.float32)
            array -= array.min()
            max_val = float(array.max())
            if max_val > 0:
                array /= max_val
            array = (array * 255.0).clip(0, 255).astype(np.uint8)

            if array.ndim == 2:
                rgb = np.stack([array, array, array], axis=-1)
            elif array.ndim == 3 and array.shape[-1] == 3:
                rgb = array
            elif array.ndim == 3 and array.shape[0] == 3:
                rgb = np.transpose(array, (1, 2, 0))
            else:
                gray = array.squeeze()
                rgb = np.stack([gray, gray, gray], axis=-1)
            return Image.fromarray(rgb).convert("RGB")

        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one sample as tensor pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_tensor_float32).

        Raises:
            IndexError: If idx is outside dataset range.
        """

        if idx < 0 or idx >= len(self.image_ids):
            raise IndexError(f"Index out of range: {idx}")

        image_id = self.image_ids[idx]
        image = self._load_image(image_id)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        labels = self.label_df.iloc[idx][self.labels].to_numpy(dtype=np.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return image_tensor, label_tensor

    def __len__(self) -> int:
        """Return the number of items.

        Args:
            None.

        Returns:
            Number of samples in this split.

        Raises:
            None.
        """

        return len(self.image_ids)


def get_transforms(train: bool, image_size: int) -> transforms.Compose:
    """Build train/eval image preprocessing transforms.

    Args:
        train: Whether to return augmentation pipeline.
        image_size: Output spatial size.

    Returns:
        A torchvision transform composition.

    Raises:
        None.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders for VinDr-CXR.

    Args:
        config: Global project config.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        ValueError: If validation split is invalid.
    """

    if not 0.0 < config.VAL_SPLIT < 1.0:
        raise ValueError("VAL_SPLIT must be in (0, 1).")

    train_tf = get_transforms(train=True, image_size=config.IMAGE_SIZE)
    eval_tf = get_transforms(train=False, image_size=config.IMAGE_SIZE)

    train_dataset_aug = VinDrCXRDataset(
        csv_path=config.TRAIN_CSV,
        img_dir=config.TRAIN_IMG_DIR,
        labels=config.LABELS,
        transform=train_tf,
        split="train",
        vote_thresh=config.MAJORITY_VOTE_THRESHOLD,
    )
    train_dataset_eval = VinDrCXRDataset(
        csv_path=config.TRAIN_CSV,
        img_dir=config.TRAIN_IMG_DIR,
        labels=config.LABELS,
        transform=eval_tf,
        split="train",
        vote_thresh=config.MAJORITY_VOTE_THRESHOLD,
    )

    n_total = len(train_dataset_aug)
    n_val = int(round(n_total * config.VAL_SPLIT))
    n_val = max(1, min(n_val, n_total - 1))
    n_train = n_total - n_val

    seed = config.SEEDS[0] if config.SEEDS else 42
    split_generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_total, generator=split_generator).tolist()
    train_indices = permutation[:n_train]
    val_indices = permutation[n_train:]

    train_subset = Subset(train_dataset_aug, train_indices)
    val_subset = Subset(train_dataset_eval, val_indices)

    test_dataset = VinDrCXRDataset(
        csv_path=config.TEST_CSV,
        img_dir=config.TEST_IMG_DIR,
        labels=config.LABELS,
        transform=eval_tf,
        split="test",
        vote_thresh=config.MAJORITY_VOTE_THRESHOLD,
    )

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return train_loader, val_loader, test_loader