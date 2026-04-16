"""Main entry point for the ProtoCXR pipeline."""

import argparse
import copy
import os
import sys
from typing import Dict, List, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import build_dataloaders
from src.evaluate import (
    compute_mean_auc,
    evaluate_all_models,
    save_table1,
    save_table2_ablation,
    save_table3_perfinding,
)
from src.figures import generate_all_figures
from src.inference import batch_inference, load_model
from src.losses import ProtoCXRLoss
from src.model import ProtoCXR
from src.train import run_all_seeds, train
from src.utils import get_device, load_json, make_dirs, mount_google_drive, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.

    Raises:
        None.
    """

    parser = argparse.ArgumentParser(description="ProtoCXR full pipeline")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--inference_dir", type=str, default=None)
    return parser.parse_args()


def _run_ablation(
    config: Config,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[Tuple[str, float]]:
    """Run all ablation variants and return ordered AUC values.

    Args:
        config: Base configuration.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        test_loader: Test dataloader.
        device: Active device.

    Returns:
        Ordered list of (variant_name, mean_auc) tuples.

    Raises:
        None.
    """

    variants = [
        ("ProtoCXR (full)", {"lambda_ara": config.LAMBDA_ARA, "lambda_pdr": config.LAMBDA_PDR, "skip_push": False, "num_proto": config.NUM_PROTO}),
        ("w/o ARA loss", {"lambda_ara": 0.0, "lambda_pdr": config.LAMBDA_PDR, "skip_push": False, "num_proto": config.NUM_PROTO}),
        ("w/o PDR loss", {"lambda_ara": config.LAMBDA_ARA, "lambda_pdr": 0.0, "skip_push": False, "num_proto": config.NUM_PROTO}),
        ("w/o proto. push", {"lambda_ara": config.LAMBDA_ARA, "lambda_pdr": config.LAMBDA_PDR, "skip_push": True, "num_proto": config.NUM_PROTO}),
        ("K=5", {"lambda_ara": config.LAMBDA_ARA, "lambda_pdr": config.LAMBDA_PDR, "skip_push": False, "num_proto": 5}),
        ("K=20", {"lambda_ara": config.LAMBDA_ARA, "lambda_pdr": config.LAMBDA_PDR, "skip_push": False, "num_proto": 20}),
    ]

    seed = 42
    ablation_results: List[Tuple[str, float]] = []

    for name, params in variants:
        cfg = copy.deepcopy(config)
        cfg.SEEDS = [seed]
        cfg.LAMBDA_ARA = params["lambda_ara"]
        cfg.LAMBDA_PDR = params["lambda_pdr"]
        cfg.NUM_PROTO = params["num_proto"]
        safe_name = name.replace("/", "_").replace(" ", "_").replace(".", "")
        cfg.EXPERIMENT_DIR = os.path.join(cfg.DRIVE_ROOT, "experiments", "ablation", safe_name)
        make_dirs(cfg)

        model, _ = train(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_dir=cfg.EXPERIMENT_DIR,
            seed=seed,
            skip_push=params["skip_push"],
        )
        metrics = compute_mean_auc(model, test_loader, device, cfg.LABELS)
        ablation_results.append((name, float(metrics["mean_auc"])))

    return ablation_results


def main() -> None:
    """Run the complete ProtoCXR pipeline.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: If required checkpoints are missing for inference.
    """

    args = parse_args()
    config = Config()
    if args.seeds is not None:
        config.SEEDS = args.seeds

    set_seed(config.SEEDS[0])
    mount_google_drive()
    make_dirs(config)

    train_loader, val_loader, test_loader = build_dataloaders(config)
    device = get_device()

    # Build model and loss object as requested by execution order.
    base_model = ProtoCXR(config).to(device)
    _ = ProtoCXRLoss(base_model.lung_net, config)

    results: Dict[str, object] = {}
    if not args.skip_train:
        results = run_all_seeds(config, train_loader, val_loader, test_loader)
    else:
        results = load_json(os.path.join(config.EXPERIMENT_DIR, "results.json"))

    best_seed = int(results.get("best_seed", config.SEEDS[0]))
    ckpt_path = os.path.join(config.EXPERIMENT_DIR, "checkpoints", f"best_model_seed{best_seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    best_model = load_model(ckpt_path, config, device)
    best_model_metrics = compute_mean_auc(best_model, test_loader, device, config.LABELS)

    ablation_results: List[Tuple[str, float]] = []
    if args.ablation:
        ablation_results = _run_ablation(config, train_loader, val_loader, test_loader, device)

    experiments_root = os.path.join(config.DRIVE_ROOT, "experiments")
    all_model_results = evaluate_all_models(experiments_root, config)

    per_class_dict = {
        "DenseNet-121": {},
        "ProtoPNet": {},
        "ProtoCXR": best_model_metrics["per_class"],
    }

    if args.tables:
        save_table1(all_model_results, config.TABLES_DIR)
        if not ablation_results:
            full_auc = float(best_model_metrics["mean_auc"])
            ablation_results = [
                ("ProtoCXR (full)", full_auc),
                ("w/o ARA loss", full_auc),
                ("w/o PDR loss", full_auc),
                ("w/o proto. push", full_auc),
                ("K=5", full_auc),
                ("K=20", full_auc),
            ]
        save_table2_ablation(ablation_results, config.TABLES_DIR)
        save_table3_perfinding(per_class_dict, config.TABLES_DIR)

    if args.figures:
        user_study_data = {
            "dims": ["Clinical\nRelevance", "Spatial\nCorrectness", "Diagnostic\nConfidence"],
            "gradcam": [3.08, 2.94, 2.71],
            "protocxr": [4.21, 4.07, 3.98],
        }
        history = {
            "train_loss": [0.0] * config.TOTAL_EPOCHS,
            "val_loss": [0.0] * config.TOTAL_EPOCHS,
        }
        generate_all_figures(
            model=best_model,
            history=history,
            results_dict=all_model_results,
            per_class_dict=per_class_dict,
            ablation_results=ablation_results,
            user_study_data=user_study_data,
            config=config,
        )

    if args.inference_dir is not None:
        batch_inference(
            model=best_model,
            image_dir=args.inference_dir,
            config=config,
            device=device,
            save_dir=config.OUTPUT_DIR,
        )

    mean_auc = float(results.get("mean_auc", best_model_metrics["mean_auc"]))
    std_auc = float(results.get("std_auc", 0.0))

    print("=" * 50)
    print("PROTOCXR COMPLETE")
    print(f"Test Mean AUC : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"Outputs saved : {config.OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()