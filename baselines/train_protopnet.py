"""
baselines/train_protopnet.py
============================
ProtoPNet baseline adapted for multi-label classification.
Trained under identical conditions to ProtoCXR, but without ARA or PDR loss.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import build_dataloaders
from src.train import run_all_seeds


def run_protopnet(dataset_name: str):
    config = Config()
    
    # Disable ARA and PDR losses for ProtoPNet baseline
    config.LAMBDA_ARA = 0.0
    config.LAMBDA_PDR = 0.0
    
    config.EXPERIMENT_DIR = os.path.join(config.DRIVE_ROOT, "experiments", "protopnet")
    
    train_loader, val_loader, test_loader = build_dataloaders(dataset_name, config, seed=42)
    
    print(f"\nTraining ProtoPNet on {dataset_name}...")
    run_all_seeds(config, train_loader, val_loader, test_loader, dataset_name, skip_push=False)


if __name__ == "__main__":
    run_protopnet("chexpert")
    run_protopnet("nih")
