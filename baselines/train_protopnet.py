"""ProtoPNet-style baseline for VinDr-CXR."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import build_dataloaders
from src.train import run_all_seeds
from src.utils import load_json, make_dirs, save_json


def main() -> None:
    """Train ProtoPNet baseline under ProtoCXR training schedule.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    config = Config()
    config.LAMBDA_ARA = 0.0
    config.LAMBDA_PDR = 0.0
    config.EXPERIMENT_DIR = os.path.join(config.DRIVE_ROOT, "experiments", "protopnet")
    make_dirs(config)

    train_loader, val_loader, test_loader = build_dataloaders(config)
    run_all_seeds(config, train_loader, val_loader, test_loader)

    results_path = os.path.join(config.EXPERIMENT_DIR, "results.json")
    payload = load_json(results_path)
    if payload:
        payload["model_name"] = "ProtoPNet"
        save_json(payload, results_path)


if __name__ == "__main__":
    main()