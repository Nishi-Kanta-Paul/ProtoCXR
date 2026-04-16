"""DenseNet-121 baseline under ProtoCXR training conditions."""

import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import build_dataloaders
from src.utils import AverageMeter, append_jsonl, get_device, make_dirs, save_json, set_seed


def _train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
) -> float:
    """Train DenseNet for one epoch.

    Args:
        model: DenseNet model.
        loader: Training dataloader.
        optimizer: Optimizer.
        criterion: BCE loss.
        device: Active device.
        config: Global config.

    Returns:
        Average training loss.

    Raises:
        None.
    """

    model.train()
    meter = AverageMeter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        meter.update(float(loss.item()), n=images.shape[0])
    return meter.avg


def _validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate DenseNet model.

    Args:
        model: DenseNet model.
        loader: Validation dataloader.
        criterion: BCE loss.
        device: Active device.

    Returns:
        Dict containing validation loss and macro AUC.

    Raises:
        None.
    """

    model.eval()
    loss_meter = AverageMeter()
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_meter.update(float(loss.item()), n=images.shape[0])
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    try:
        auc = float(roc_auc_score(labels_np, probs, average="macro"))
    except ValueError:
        auc = float("nan")
    return {"loss": loss_meter.avg, "auc": auc}


def _test_auc(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute macro test AUC.

    Args:
        model: Trained model.
        loader: Test dataloader.
        device: Active device.

    Returns:
        Macro ROC-AUC.

    Raises:
        None.
    """

    model.eval()
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    try:
        return float(roc_auc_score(labels_np, probs, average="macro"))
    except ValueError:
        return float("nan")


def main() -> None:
    """Run DenseNet-121 baseline across configured seeds.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    config = Config()
    device = get_device()
    make_dirs(config)

    exp_dir = os.path.join(config.DRIVE_ROOT, "experiments", "densenet121")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(config)
    per_seed_auc: Dict[str, float] = {}

    for seed in config.SEEDS:
        set_seed(seed)
        model = timm.create_model(
            "densenet121",
            pretrained=True,
            num_classes=config.NUM_CLASSES,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TOTAL_EPOCHS)

        log_path = os.path.join(log_dir, f"train_log_seed{seed}.jsonl")
        ckpt_path = os.path.join(ckpt_dir, f"best_model_seed{seed}.pt")
        if os.path.exists(log_path):
            os.remove(log_path)

        best_val_auc = float("-inf")
        best_state = None

        for epoch in range(1, config.TOTAL_EPOCHS + 1):
            train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, config)
            val_stats = _validate(model, val_loader, criterion, device)
            scheduler.step()

            append_jsonl(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_stats["loss"],
                    "val_auc": val_stats["auc"],
                    "timestamp": datetime.utcnow().isoformat(),
                },
                log_path,
            )

            if val_stats["auc"] > best_val_auc:
                best_val_auc = val_stats["auc"]
                best_state = model.state_dict()
                torch.save({"epoch": epoch, "model_state": best_state, "val_auc": best_val_auc}, ckpt_path)

        if best_state is not None:
            model.load_state_dict(best_state)
        per_seed_auc[str(seed)] = _test_auc(model, test_loader, device)

    auc_values = list(per_seed_auc.values())
    best_seed = max(per_seed_auc, key=per_seed_auc.get)
    results = {
        "model_name": "DenseNet-121",
        "dataset": "VinDr-CXR",
        "mean_auc": float(np.mean(auc_values)),
        "std_auc": float(np.std(auc_values)),
        "per_seed_auc": per_seed_auc,
        "best_seed": int(best_seed),
        "config_snapshot": vars(config),
        "timestamp": datetime.utcnow().isoformat(),
    }
    save_json(results, os.path.join(exp_dir, "results.json"))


if __name__ == "__main__":
    main()