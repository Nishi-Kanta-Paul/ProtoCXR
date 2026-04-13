"""
src/train.py
============
Complete 4-phase training loop for ProtoCXR:

  Phase 1 — Warm-up        : Only prototype layer + FC trained.
  Phase 2 — Joint Training  : All layers trained with differential LRs.
  Phase 3 — Prototype Push  : Periodically replace prototypes with real patches.
  Phase 4 — FC Fine-tuning  : Only the final FC layer trained.
"""

import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.losses import ProtoCXRLoss
from src.model import LungMaskNet, ProtoCXR
from src.utils import AverageMeter, append_jsonl, save_json, set_seed


# ─── Single Epoch ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ProtoCXR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ProtoCXRLoss,
    device: torch.device,
    epoch: int,
    config: Config,
) -> Dict[str, float]:
    """Train the model for a single epoch.

    Runs forward/backward passes for every mini-batch in ``loader``,
    clips gradients, updates parameters, and tracks losses via
    :class:`~src.utils.AverageMeter`.

    Args:
        model:     :class:`~src.model.ProtoCXR` model in training mode.
        loader:    Training :class:`~torch.utils.data.DataLoader`.
        optimizer: PyTorch optimizer.
        loss_fn:   :class:`~src.losses.ProtoCXRLoss` composite loss.
        device:    Compute device.
        epoch:     Current epoch number (1-indexed, for display).
        config:    ``Config`` instance (reads ``GRAD_CLIP``).

    Returns:
        Dictionary mapping loss component names to their epoch-average
        float values: ``{"total", "bce", "ara", "pdr", "sep"}``.
    """
    model.train()
    meters = {k: AverageMeter() for k in ("total", "bce", "ara", "pdr", "sep")}

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, sim_maps, _ = model(images, return_sim_maps=True)  # type: ignore[misc]
        loss_dict = loss_fn(
            logits, labels, sim_maps, images, model.prototypes
        )

        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=config.GRAD_CLIP,
        )
        optimizer.step()

        bsz = images.size(0)
        for k, meter in meters.items():
            meter.update(loss_dict[k].item(), n=bsz)

        pbar.set_postfix(
            loss=f"{meters['total'].avg:.4f}",
            bce=f"{meters['bce'].avg:.4f}",
        )

    return {k: m.avg for k, m in meters.items()}


# ─── Validation ───────────────────────────────────────────────────────────────

def validate(
    model: ProtoCXR,
    loader: DataLoader,
    loss_fn: ProtoCXRLoss,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation or test set.

    Collects all logits and labels over the full loader, computes
    per-sample BCE loss, and macro-averaged ROC-AUC.

    Args:
        model:    :class:`~src.model.ProtoCXR` model.
        loader:   Validation :class:`~torch.utils.data.DataLoader`.
        loss_fn:  :class:`~src.losses.ProtoCXRLoss` for loss computation.
        device:   Compute device.

    Returns:
        Tuple ``(avg_val_loss, mean_auc)``:
          - ``avg_val_loss`` (float): Mean total loss over the loader.
          - ``mean_auc`` (float): Macro-averaged ROC-AUC score.
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = AverageMeter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, sim_maps, _ = model(images, return_sim_maps=True)  # type: ignore[misc]
            loss_dict = loss_fn(
                logits, labels, sim_maps, images, model.prototypes
            )
            total_loss.update(loss_dict["total"].item(), n=images.size(0))

            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    preds  = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_labels, dim=0).numpy()

    try:
        mean_auc = roc_auc_score(targets, preds, average="macro")
    except ValueError:
        mean_auc = float("nan")

    return total_loss.avg, float(mean_auc)


# ─── Main Training Function ───────────────────────────────────────────────────

def train(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_dir: str,
    seed: int,
    skip_push: bool = False,
) -> Tuple[ProtoCXR, Dict[str, List]]:
    """Run the full 4-phase ProtoCXR training for a single seed.

    Phases:
        1. **Warm-up** (epochs 1 – ``WARMUP_EPOCHS``):
           Backbone frozen; prototype layer + FC trained.
        2. **Joint Training** (epochs ``WARMUP_EPOCHS+1`` – ``WARMUP_EPOCHS+JOINT_EPOCHS``):
           All layers trained with differential learning rates.
        3. **Prototype Push** (every ``PUSH_EVERY`` epochs within Phase 2):
           Prototypes replaced with nearest real training patches.
        4. **FC Fine-tuning** (last ``FINETUNE_EPOCHS`` epochs):
           Backbone and prototypes frozen; only FC weights trained.

    Logging:
        - Appends one JSON line per epoch to
          ``<experiment_dir>/logs/train_log_seed<seed>.jsonl``.
        - Saves best checkpoint to
          ``<experiment_dir>/checkpoints/best_model_seed<seed>.pt``
          whenever validation AUC improves.

    Args:
        config:         ``Config`` instance.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        experiment_dir: Root directory for checkpoints and logs.
        seed:           Random seed (used for filenames and reproducibility).
        skip_push:      If ``True``, skip prototype push (ablation variant).

    Returns:
        Tuple ``(best_model, history)`` where:
          - ``best_model`` is the :class:`~src.model.ProtoCXR` instance
            loaded with best-validation-AUC weights.
          - ``history`` is a dict with lists ``"train_loss"``, ``"val_loss"``,
            ``"val_auc"`` indexed by epoch.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build model + loss ────────────────────────────────────────────────────
    model = ProtoCXR(
        num_classes=config.NUM_CLASSES,
        num_proto=config.NUM_PROTO,
        feat_dim=config.FEAT_DIM,
        backbone_name=config.BACKBONE,
        backbone_pretrained=config.BACKBONE_PRETRAINED,
        sim_epsilon=config.SIM_EPSILON,
    ).to(device)

    loss_fn = ProtoCXRLoss(
        lung_net=model.lung_net,
        lambda_ara=config.LAMBDA_ARA,
        lambda_pdr=config.LAMBDA_PDR,
        lambda_sep=config.LAMBDA_SEP,
        sigma=config.PDR_SIGMA,
        num_classes=config.NUM_CLASSES,
        num_proto=config.NUM_PROTO,
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    log_dir  = os.path.join(experiment_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, f"best_model_seed{seed}.pt")
    log_path       = os.path.join(log_dir,   f"train_log_seed{seed}.jsonl")

    # ── History ───────────────────────────────────────────────────────────────
    history: Dict[str, List] = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_auc = float("-inf")
    best_model_state: Optional[Dict] = None

    # ── Phase boundaries ──────────────────────────────────────────────────────
    p1_end = config.WARMUP_EPOCHS
    p2_end = config.WARMUP_EPOCHS + config.JOINT_EPOCHS
    p4_end = p2_end + config.FINETUNE_EPOCHS   # == TOTAL_EPOCHS

    # ========================================================================
    # Phase 1 — Warm-up
    # ========================================================================
    print(f"\n{'═' * 60}")
    print(f"  Phase 1: Warm-up — epochs 1–{p1_end} | seed {seed}")
    print(f"{'═' * 60}")

    model.freeze_backbone(freeze=True)
    model.freeze_prototypes(freeze=False)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.prototypes, "lr": config.LR_PROTO},
            {"params": model.fc.parameters(), "lr": config.LR_FC},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(1, p1_end + 1):
        train_losses = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, config
        )
        val_loss, val_auc = validate(model, val_loader, loss_fn, device)
        _log_and_checkpoint(
            epoch, train_losses, val_loss, val_auc, optimizer,
            "warmup", log_path, best_ckpt_path, model,
            history, best_val_auc,
        )
        best_val_auc, best_model_state = _update_best(
            val_auc, best_val_auc, model, optimizer, epoch, best_ckpt_path
        )

    # ========================================================================
    # Phase 2 — Joint Training  +  Phase 3 — Prototype Push
    # ========================================================================
    print(f"\n{'═' * 60}")
    print(f"  Phase 2: Joint Training — epochs {p1_end + 1}–{p2_end} | seed {seed}")
    print(f"{'═' * 60}")

    model.freeze_backbone(freeze=False)
    model.freeze_prototypes(freeze=False)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE},
            {"params": model.proj.parameters(),     "lr": config.LR_BACKBONE},
            {"params": model.prototypes,            "lr": config.LR_PROTO},
            {"params": model.fc.parameters(),       "lr": config.LR_FC},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.JOINT_EPOCHS
    )

    for epoch in range(p1_end + 1, p2_end + 1):
        train_losses = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, config
        )
        val_loss, val_auc = validate(model, val_loader, loss_fn, device)
        _log_and_checkpoint(
            epoch, train_losses, val_loss, val_auc, optimizer,
            "joint", log_path, best_ckpt_path, model,
            history, best_val_auc,
        )
        best_val_auc, best_model_state = _update_best(
            val_auc, best_val_auc, model, optimizer, epoch, best_ckpt_path
        )
        scheduler.step()

        # Phase 3 — Prototype Push every PUSH_EVERY epochs
        relative_epoch = epoch - p1_end
        if not skip_push and relative_epoch % config.PUSH_EVERY == 0:
            print(f"  → Prototype push at epoch {epoch} …")
            model.push_prototypes(train_loader, device)

    # ========================================================================
    # Phase 4 — FC Fine-tuning
    # ========================================================================
    print(f"\n{'═' * 60}")
    print(f"  Phase 4: FC Fine-tuning — epochs {p2_end + 1}–{p4_end} | seed {seed}")
    print(f"{'═' * 60}")

    model.freeze_backbone(freeze=True)
    model.freeze_prototypes(freeze=True)

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=config.LR_FC,
        weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(p2_end + 1, p4_end + 1):
        train_losses = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, config
        )
        val_loss, val_auc = validate(model, val_loader, loss_fn, device)
        _log_and_checkpoint(
            epoch, train_losses, val_loss, val_auc, optimizer,
            "finetune", log_path, best_ckpt_path, model,
            history, best_val_auc,
        )
        best_val_auc, best_model_state = _update_best(
            val_auc, best_val_auc, model, optimizer, epoch, best_ckpt_path
        )

    # ── Load best checkpoint ─────────────────────────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n  Training complete — Best val AUC: {best_val_auc:.4f}")
    return model, history


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _log_and_checkpoint(
    epoch: int,
    train_losses: Dict[str, float],
    val_loss: float,
    val_auc: float,
    optimizer: torch.optim.Optimizer,
    phase: str,
    log_path: str,
    ckpt_path: str,
    model: ProtoCXR,
    history: Dict[str, List],
    best_val_auc: float,
) -> None:
    """Append one JSON-L log line and print epoch summary.

    Args:
        epoch:         Current epoch number.
        train_losses:  Dict returned by :func:`train_one_epoch`.
        val_loss:      Validation total loss.
        val_auc:       Validation macro-AUC.
        optimizer:     Current optimizer (for LR extraction).
        phase:         Phase name string for the log record.
        log_path:      Path to the ``.jsonl`` log file.
        ckpt_path:     Path to the best checkpoint file.
        model:         Model instance.
        history:       Mutable history dict updated in-place.
        best_val_auc:  Current best AUC (for log purposes).
    """
    history["train_loss"].append(train_losses["total"])
    history["val_loss"].append(val_loss)
    history["val_auc"].append(val_auc)

    lr = optimizer.param_groups[0]["lr"]
    record = {
        "epoch":      epoch,
        "train_loss": round(train_losses["total"], 6),
        "val_loss":   round(val_loss, 6),
        "val_auc":    round(val_auc, 6),
        "bce":        round(train_losses["bce"], 6),
        "ara":        round(train_losses["ara"], 6),
        "pdr":        round(train_losses["pdr"], 6),
        "sep":        round(train_losses["sep"], 6),
        "lr":         lr,
        "phase":      phase,
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
    }
    append_jsonl(record, log_path)

    star = " ★" if val_auc > best_val_auc else ""
    print(
        f"  [Epoch {epoch:3d}] train_loss={train_losses['total']:.4f} "
        f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}{star}"
    )


def _update_best(
    val_auc: float,
    best_val_auc: float,
    model: ProtoCXR,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_path: str,
) -> Tuple[float, Optional[Dict]]:
    """Save checkpoint if current AUC is best so far.

    Args:
        val_auc:      Current epoch's validation AUC.
        best_val_auc: Previous best AUC.
        model:        Model to checkpoint.
        optimizer:    Optimizer state to save.
        epoch:        Current epoch number.
        ckpt_path:    Destination file path.

    Returns:
        Tuple ``(new_best_auc, model_state_dict)`` where
        ``model_state_dict`` is the saved state (or ``None`` if not improved).
    """
    if val_auc > best_val_auc:
        state = model.state_dict()
        torch.save(
            {
                "epoch":             epoch,
                "model_state_dict":  state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc":           val_auc,
            },
            ckpt_path,
        )
        return val_auc, dict(state)
    return best_val_auc, None


# ─── Multi-Seed Runner ────────────────────────────────────────────────────────

def run_all_seeds(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset_name: str,
    skip_push: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Train ProtoCXR across all configured seeds and aggregate results.

    Each seed is trained independently with :func:`train`. The aggregated
    mean / std AUC and a config snapshot are saved to
    ``<EXPERIMENT_DIR>/results.json``.

    Args:
        config:       ``Config`` instance.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        test_loader:  Test DataLoader (not used in this function but
                      reserved for the caller to run :func:`~src.evaluate.compute_mean_auc`).
        dataset_name: ``"chexpert"`` or ``"nih"``.
        skip_push:    If ``True``, skip prototype push in all seeds.

    Returns:
        Dictionary ``{seed: {"val_auc": float, "history": dict}}`` for all
        seeds in ``config.SEEDS``.
    """
    all_results: Dict[int, Dict[str, Any]] = {}

    for seed in config.SEEDS:
        print(f"\n{'═' * 70}")
        print(f"  SEED {seed}  |  dataset: {dataset_name}")
        print(f"{'═' * 70}")
        set_seed(seed)

        best_model, history = train(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_dir=config.EXPERIMENT_DIR,
            seed=seed,
            skip_push=skip_push,
        )

        final_val_auc = max(history["val_auc"]) if history["val_auc"] else float("nan")
        all_results[seed] = {"val_auc": final_val_auc, "history": history}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aucs = [v["val_auc"] for v in all_results.values()]
    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))

    best_seed = max(all_results, key=lambda s: all_results[s]["val_auc"])

    results_payload: Dict[str, Any] = {
        "model_name":    "ProtoCXR",
        "dataset":       dataset_name,
        "mean_auc":      round(mean_auc, 6),
        "std_auc":       round(std_auc, 6),
        "per_seed_auc":  {str(s): round(v["val_auc"], 6) for s, v in all_results.items()},
        "best_seed":     best_seed,
        "config_snapshot": {
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_")
        },
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    results_path = os.path.join(config.EXPERIMENT_DIR, "results.json")
    save_json(results_payload, results_path)
    print(f"\n  Results saved → {results_path}")
    print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    return all_results
