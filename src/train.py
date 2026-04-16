"""Training loops for ProtoCXR."""

import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.losses import ProtoCXRLoss
from src.model import ProtoCXR
from src.utils import AverageMeter, append_jsonl, save_json, set_seed


def train_one_epoch(
    model: ProtoCXR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ProtoCXRLoss,
    device: torch.device,
    epoch: int,
    config: Config,
) -> Dict[str, float]:
    """Train ProtoCXR for one epoch.

    Args:
        model: ProtoCXR model in training mode.
        loader: Training dataloader.
        optimizer: Optimizer for the current phase.
        loss_fn: Combined ProtoCXR loss module.
        device: Target device.
        epoch: Current epoch number (1-indexed).
        config: Global configuration.

    Returns:
        Dictionary containing averaged values for total, bce, ara, pdr, sep.

    Raises:
        None.
    """

    model.train()
    meters = {name: AverageMeter() for name in ["total", "bce", "ara", "pdr", "sep"]}
    progress = tqdm(loader, leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, sim_maps, _ = model(images, return_sim_maps=True)
        loss_dict = loss_fn(logits, labels, sim_maps, images, model.prototypes)
        loss_dict["total"].backward()

        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            config.GRAD_CLIP,
        )
        optimizer.step()

        batch_size = images.shape[0]
        for key in meters:
            meters[key].update(float(loss_dict[key].item()), n=batch_size)

        progress.set_description(f"Epoch {epoch} | Loss: {meters['total'].avg:.4f}")

    return {key: meter.avg for key, meter in meters.items()}


def validate(
    model: ProtoCXR,
    loader: DataLoader,
    loss_fn: ProtoCXRLoss,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate ProtoCXR with loss and macro AUC.

    Args:
        model: ProtoCXR model.
        loader: Validation dataloader.
        loss_fn: Combined ProtoCXR loss module.
        device: Target device.

    Returns:
        Tuple of (average validation loss, macro mean AUC).

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

            logits, sim_maps, _ = model(images, return_sim_maps=True)
            losses = loss_fn(logits, labels, sim_maps, images, model.prototypes)
            loss_meter.update(float(losses["total"].item()), n=images.shape[0])

            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    try:
        mean_auc = float(roc_auc_score(labels_np, probs, average="macro"))
    except ValueError:
        mean_auc = float("nan")

    return loss_meter.avg, mean_auc


def _build_optimizer_for_phase(
    model: ProtoCXR,
    config: Config,
    phase: str,
) -> torch.optim.Optimizer:
    """Create optimizer matching the current training phase.

    Args:
        model: ProtoCXR model.
        config: Global config.
        phase: One of warmup, joint, finetune.

    Returns:
        Configured AdamW optimizer.

    Raises:
        ValueError: If phase name is unknown.
    """

    if phase == "warmup":
        params = [
            {"params": [model.prototypes], "lr": config.LR_PROTO, "name": "proto"},
            {"params": model.fc.parameters(), "lr": config.LR_FC, "name": "fc"},
        ]
    elif phase == "joint":
        params = [
            {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE, "name": "backbone"},
            {"params": model.proj.parameters(), "lr": config.LR_BACKBONE, "name": "backbone"},
            {"params": [model.prototypes], "lr": config.LR_PROTO, "name": "proto"},
            {"params": model.fc.parameters(), "lr": config.LR_FC, "name": "fc"},
        ]
    elif phase == "finetune":
        params = [
            {"params": model.fc.parameters(), "lr": config.LR_FC, "name": "fc"},
        ]
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return torch.optim.AdamW(params, weight_decay=config.WEIGHT_DECAY)


def _extract_lr(optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """Extract backbone and prototype learning rates from optimizer.

    Args:
        optimizer: Active optimizer.

    Returns:
        Tuple of (lr_backbone, lr_proto).

    Raises:
        None.
    """

    lr_backbone = 0.0
    lr_proto = 0.0
    for group in optimizer.param_groups:
        group_name = group.get("name", "")
        if group_name == "backbone":
            lr_backbone = float(group["lr"])
        if group_name == "proto":
            lr_proto = float(group["lr"])
    return lr_backbone, lr_proto


def _print_phase_banner(index: int, name: str, start_epoch: int, end_epoch: int) -> None:
    """Print standardized phase transition banner.

    Args:
        index: Phase number.
        name: Phase name.
        start_epoch: Start epoch index.
        end_epoch: End epoch index.

    Returns:
        None.

    Raises:
        None.
    """

    print("=" * 60)
    print(f"Phase {index}: {name} - epochs {start_epoch}-{end_epoch}")
    print("=" * 60)


def train(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_dir: str,
    seed: int,
    skip_push: bool = False,
) -> Tuple[ProtoCXR, Dict[str, List[float]]]:
    """Run the full four-phase ProtoCXR training for one seed.

    Args:
        config: Global config.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        experiment_dir: Path where logs/checkpoints are stored.
        seed: Random seed for this run.
        skip_push: If True, skips prototype push during joint phase.

    Returns:
        Tuple of (best_model, history_dict).

    Raises:
        RuntimeError: If no checkpoint is produced.
    """

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    logs_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ckpt_path = os.path.join(checkpoints_dir, f"best_model_seed{seed}.pt")
    log_path = os.path.join(logs_dir, f"train_log_seed{seed}.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    model = ProtoCXR(config).to(device)
    loss_fn = ProtoCXRLoss(model.lung_net, config)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "loss_bce": [],
        "loss_ara": [],
        "loss_pdr": [],
        "loss_sep": [],
    }

    best_auc = float("-inf")
    best_state: Dict[str, torch.Tensor] = {}

    warm_start, warm_end = 1, config.WARMUP_EPOCHS
    joint_start, joint_end = warm_end + 1, warm_end + config.JOINT_EPOCHS
    fine_start, fine_end = joint_end + 1, joint_end + config.FINETUNE_EPOCHS

    # Phase 1: Warm-up
    _print_phase_banner(1, "Warm-up", warm_start, warm_end)
    model.freeze_backbone(True)
    model.freeze_prototypes(False)
    optimizer = _build_optimizer_for_phase(model, config, phase="warmup")

    for epoch in range(warm_start, warm_end + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config)
        val_loss, val_auc = validate(model, val_loader, loss_fn, device)
        lr_backbone, lr_proto = _extract_lr(optimizer)

        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["loss_bce"].append(train_losses["bce"])
        history["loss_ara"].append(train_losses["ara"])
        history["loss_pdr"].append(train_losses["pdr"])
        history["loss_sep"].append(train_losses["sep"])

        append_jsonl(
            {
                "epoch": epoch,
                "phase": "warmup",
                "train_loss": train_losses["total"],
                "val_loss": val_loss,
                "val_auc": val_auc,
                "loss_bce": train_losses["bce"],
                "loss_ara": train_losses["ara"],
                "loss_pdr": train_losses["pdr"],
                "loss_sep": train_losses["sep"],
                "lr_backbone": lr_backbone,
                "lr_proto": lr_proto,
                "timestamp": datetime.utcnow().isoformat(),
            },
            log_path,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(config),
                },
                ckpt_path,
            )

    # Phase 2 + push
    _print_phase_banner(2, "Joint Training", joint_start, joint_end)
    model.freeze_backbone(False)
    model.freeze_prototypes(False)
    optimizer = _build_optimizer_for_phase(model, config, phase="joint")

    for epoch in range(joint_start, joint_end + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config)
        val_loss, val_auc = validate(model, val_loader, loss_fn, device)

        if not skip_push and epoch % config.PUSH_EVERY == 0:
            model.push_prototypes(train_loader, device)

        lr_backbone, lr_proto = _extract_lr(optimizer)
        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["loss_bce"].append(train_losses["bce"])
        history["loss_ara"].append(train_losses["ara"])
        history["loss_pdr"].append(train_losses["pdr"])
        history["loss_sep"].append(train_losses["sep"])

        append_jsonl(
            {
                "epoch": epoch,
                "phase": "joint",
                "train_loss": train_losses["total"],
                "val_loss": val_loss,
                "val_auc": val_auc,
                "loss_bce": train_losses["bce"],
                "loss_ara": train_losses["ara"],
                "loss_pdr": train_losses["pdr"],
                "loss_sep": train_losses["sep"],
                "lr_backbone": lr_backbone,
                "lr_proto": lr_proto,
                "timestamp": datetime.utcnow().isoformat(),
            },
            log_path,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(config),
                },
                ckpt_path,
            )

    # Phase 4: FC fine-tuning
    _print_phase_banner(4, "FC Fine-tune", fine_start, fine_end)
    model.freeze_backbone(True)
    model.freeze_prototypes(True)
    optimizer = _build_optimizer_for_phase(model, config, phase="finetune")

    for epoch in range(fine_start, fine_end + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config)
        val_loss, val_auc = validate(model, val_loader, loss_fn, device)
        lr_backbone, lr_proto = _extract_lr(optimizer)

        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["loss_bce"].append(train_losses["bce"])
        history["loss_ara"].append(train_losses["ara"])
        history["loss_pdr"].append(train_losses["pdr"])
        history["loss_sep"].append(train_losses["sep"])

        append_jsonl(
            {
                "epoch": epoch,
                "phase": "finetune",
                "train_loss": train_losses["total"],
                "val_loss": val_loss,
                "val_auc": val_auc,
                "loss_bce": train_losses["bce"],
                "loss_ara": train_losses["ara"],
                "loss_pdr": train_losses["pdr"],
                "loss_sep": train_losses["sep"],
                "lr_backbone": lr_backbone,
                "lr_proto": lr_proto,
                "timestamp": datetime.utcnow().isoformat(),
            },
            log_path,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(config),
                },
                ckpt_path,
            )

    if not best_state:
        raise RuntimeError("No best checkpoint state was captured during training.")

    model.load_state_dict(best_state)
    return model, history


def _evaluate_test_auc(model: ProtoCXR, loader: DataLoader, device: torch.device) -> float:
    """Compute macro ROC-AUC on test loader.

    Args:
        model: Trained model in eval mode.
        loader: Test dataloader.
        device: Active device.

    Returns:
        Macro ROC-AUC value.

    Raises:
        None.
    """

    model.eval()
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits, _ = model(images, return_sim_maps=False)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs_np = torch.cat(all_probs, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    try:
        return float(roc_auc_score(labels_np, probs_np, average="macro"))
    except ValueError:
        return float("nan")


def run_all_seeds(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    skip_push: bool = False,
) -> Dict[str, object]:
    """Train ProtoCXR over all configured seeds and save aggregate results.

    Args:
        config: Global config.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        test_loader: Test dataloader.
        skip_push: Whether to disable prototype push.

    Returns:
        Aggregated result dict saved to results.json.

    Raises:
        None.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed_auc: Dict[str, float] = {}
    best_seed = config.SEEDS[0]
    best_auc = float("-inf")

    for seed in config.SEEDS:
        model, _ = train(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_dir=config.EXPERIMENT_DIR,
            seed=seed,
            skip_push=skip_push,
        )
        seed_auc = _evaluate_test_auc(model, test_loader, device)
        per_seed_auc[str(seed)] = seed_auc
        if seed_auc > best_auc:
            best_auc = seed_auc
            best_seed = seed

    auc_values = list(per_seed_auc.values())
    results = {
        "model_name": "ProtoCXR",
        "dataset": "VinDr-CXR",
        "mean_auc": float(np.mean(auc_values)),
        "std_auc": float(np.std(auc_values)),
        "per_seed_auc": per_seed_auc,
        "best_seed": int(best_seed),
        "config_snapshot": vars(config),
        "timestamp": datetime.utcnow().isoformat(),
    }

    save_json(results, os.path.join(config.EXPERIMENT_DIR, "results.json"))
    return results