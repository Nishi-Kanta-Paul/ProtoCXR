"""
baselines/train_densenet121.py
==============================
DenseNet-121 baseline trained under identical conditions to ProtoCXR.
"""

import datetime
import os
from typing import Any, Dict, List

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import build_dataloaders
from src.utils import AverageMeter, set_seed, get_device, make_dirs, save_json, append_jsonl


def train_one_epoch(model, loader, optimizer, loss_fn, device, config):
    model.train()
    meter = AverageMeter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
        optimizer.step()
        
        meter.update(loss.item(), n=images.size(0))
        
    return meter.avg


def validate(model, loader, loss_fn, device):
    model.eval()
    meter = AverageMeter()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)
            
            meter.update(loss.item(), n=images.size(0))
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_labels).numpy()
    
    try:
        auc = roc_auc_score(targets, preds, average="macro")
    except ValueError:
        auc = float("nan")
        
    return meter.avg, float(auc)


def run_densenet(config: Config, dataset_name: str, device: torch.device):
    exp_dir = os.path.join(config.DRIVE_ROOT, "experiments", "densenet121")
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    train_loader, val_loader, test_loader = build_dataloaders(dataset_name, config, seed=42)
    
    all_results = {}
    
    for seed in config.SEEDS:
        set_seed(seed)
        print(f"\n--- DenseNet-121 | Dataset: {dataset_name} | Seed: {seed} ---")
        
        model = timm.create_model("densenet121", pretrained=True, num_classes=config.NUM_CLASSES)
        model = model.to(device)
        
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TOTAL_EPOCHS)
        
        best_auc = -1.0
        log_path = os.path.join(exp_dir, "logs", f"train_log_seed{seed}.jsonl")
        ckpt_path = os.path.join(exp_dir, "checkpoints", f"best_model_seed{seed}.pt")
        
        for epoch in range(1, config.TOTAL_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config)
            val_loss, val_auc = validate(model, val_loader, loss_fn, device)
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), ckpt_path)
                star = " ★"
            else:
                star = ""
                
            print(f"Epoch {epoch:2d}/{config.TOTAL_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | AUC: {val_auc:.4f}{star}")
            
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            append_jsonl(record, log_path)
            scheduler.step()
            
        all_results[seed] = best_auc

    aucs = list(all_results.values())
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    best_seed = max(all_results, key=all_results.get)
    
    res = {
        "model_name": "DenseNet-121",
        "dataset": dataset_name,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "per_seed_auc": {str(k): v for k, v in all_results.items()},
        "best_seed": best_seed,
        "config_snapshot": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    save_json(res, os.path.join(exp_dir, "results.json"))


if __name__ == "__main__":
    config = Config()
    device = get_device()
    run_densenet(config, "chexpert", device)
    run_densenet(config, "nih", device)
