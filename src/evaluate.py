"""Evaluation and table generation utilities for ProtoCXR."""

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.utils import load_json


def compute_mean_auc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: List[str],
) -> Dict[str, object]:
    """Compute macro and per-class ROC-AUC.

    Args:
        model: Trained model.
        loader: Evaluation dataloader.
        device: Active device.
        label_names: Ordered list of class names.

    Returns:
        Dict containing mean_auc and per_class values.

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
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    per_class: Dict[str, float] = {}
    valid_aucs: List[float] = []
    for idx, label in enumerate(label_names):
        label_gt = labels[:, idx]
        label_pred = probs[:, idx]
        if len(np.unique(label_gt)) < 2:
            continue
        auc = float(roc_auc_score(label_gt, label_pred))
        per_class[label] = auc
        valid_aucs.append(auc)

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return {"mean_auc": mean_auc, "per_class": per_class}


def evaluate_all_models(experiments_dir: str, config: object) -> Dict[str, Dict[str, object]]:
    """Load mean AUC for all available experiment result files.

    Args:
        experiments_dir: Root experiments directory.
        config: Config object (unused, kept for API consistency).

    Returns:
        Dictionary keyed by model name.

    Raises:
        None.
    """

    del config
    results: Dict[str, Dict[str, object]] = {
        "DenseNet-121": {"mean_auc": float("nan"), "interpretable": False},
        "ProtoPNet": {"mean_auc": float("nan"), "interpretable": True},
        "CBM": {"mean_auc": float("nan"), "interpretable": True},
        "ProtoCXR": {"mean_auc": float("nan"), "interpretable": True},
    }

    path_map = {
        "DenseNet-121": os.path.join(experiments_dir, "densenet121", "results.json"),
        "ProtoPNet": os.path.join(experiments_dir, "protopnet", "results.json"),
        "CBM": os.path.join(experiments_dir, "cbm", "results.json"),
        "ProtoCXR": os.path.join(experiments_dir, "protocxr", "results.json"),
    }

    for model_name, path in path_map.items():
        payload = load_json(path)
        if payload:
            results[model_name]["mean_auc"] = float(payload.get("mean_auc", float("nan")))

    # Backward-compatible fallback scan.
    discovered = glob.glob(os.path.join(experiments_dir, "*", "results.json"))
    for path in discovered:
        payload = load_json(path)
        model_name = str(payload.get("model_name", ""))
        if model_name in results and "mean_auc" in payload:
            results[model_name]["mean_auc"] = float(payload["mean_auc"])

    return results


def save_table1(results_dict: Dict[str, Dict[str, object]], tables_dir: str) -> None:
    """Save TABLE I mean AUC comparison in CSV and IEEE-style TXT.

    Args:
        results_dict: Dict from evaluate_all_models.
        tables_dir: Output directory for table files.

    Returns:
        None.

    Raises:
        OSError: If writing table files fails.
    """

    os.makedirs(tables_dir, exist_ok=True)

    dense_auc = float(results_dict.get("DenseNet-121", {}).get("mean_auc", float("nan")))
    rows = [
        {"Method": "DenseNet-121 [11]", "Interp.": "No", "Mean AUC": dense_auc},
        {"Method": "Grad-CAM [4]", "Interp.": "Post-hoc", "Mean AUC": dense_auc},
        {
            "Method": "CBM [14]",
            "Interp.": "Yes",
            "Mean AUC": float(results_dict.get("CBM", {}).get("mean_auc", float("nan"))),
        },
        {
            "Method": "ProtoPNet [15]",
            "Interp.": "Yes",
            "Mean AUC": float(results_dict.get("ProtoPNet", {}).get("mean_auc", float("nan"))),
        },
        {"Method": "ProtoTree [16]", "Interp.": "Yes", "Mean AUC": 0.843},
        {
            "Method": "ProtoCXR (ours)",
            "Interp.": "Yes",
            "Mean AUC": float(results_dict.get("ProtoCXR", {}).get("mean_auc", float("nan"))),
        },
    ]

    df = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, "table1_auc.csv")
    txt_path = os.path.join(tables_dir, "table1_auc.txt")
    df.to_csv(csv_path, index=False)

    lines = [
        "=" * 46,
        "TABLE I - MEAN AUC - VINDR-CXR",
        "=" * 46,
        f"{'Method':<24} {'Interp.':<10} {'Mean AUC':>8}",
        "-" * 46,
    ]
    for row in rows:
        value = row["Mean AUC"]
        value_str = f"{value:.3f}" if np.isfinite(value) else "N/A"
        lines.append(f"{row['Method']:<24} {row['Interp.']:<10} {value_str:>8}")
    lines.append("=" * 46)

    with open(txt_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def save_table2_ablation(ablation_results: List[Tuple[str, float]], tables_dir: str) -> None:
    """Save TABLE II ablation results in CSV and TXT formats.

    Args:
        ablation_results: Ordered list of (name, auc) tuples.
        tables_dir: Output table directory.

    Returns:
        None.

    Raises:
        ValueError: If full model entry is missing.
    """

    os.makedirs(tables_dir, exist_ok=True)
    full_auc = None
    for name, auc in ablation_results:
        if name == "ProtoCXR (full)":
            full_auc = float(auc)
            break
    if full_auc is None:
        raise ValueError("Ablation results must include 'ProtoCXR (full)'.")

    rows = []
    for name, auc in ablation_results:
        rows.append({
            "Configuration": name,
            "Mean AUC": float(auc),
            "Delta AUC": float(auc) - full_auc,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, "table2_ablation.csv")
    txt_path = os.path.join(tables_dir, "table2_ablation.txt")
    df.to_csv(csv_path, index=False)

    lines = [
        "=" * 62,
        "TABLE II - ABLATION STUDY",
        "=" * 62,
        f"{'Configuration':<28} {'AUC':>8} {'Delta AUC':>12}",
        "-" * 62,
    ]
    for row in rows:
        lines.append(
            f"{row['Configuration']:<28} {row['Mean AUC']:>8.3f} {row['Delta AUC']:>12.3f}"
        )
    lines.append("=" * 62)

    with open(txt_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def save_table3_perfinding(per_class_dict: Dict[str, Dict[str, float]], tables_dir: str) -> None:
    """Save TABLE III per-finding AUC values.

    Args:
        per_class_dict: Dict mapping model names to per-class AUC dicts.
        tables_dir: Output table directory.

    Returns:
        None.

    Raises:
        OSError: If writing files fails.
    """

    os.makedirs(tables_dir, exist_ok=True)
    models = ["DenseNet-121", "ProtoPNet", "ProtoCXR"]

    labels = []
    for model_name in models:
        labels.extend(list(per_class_dict.get(model_name, {}).keys()))
    labels = sorted(set(labels))

    rows = []
    for label in labels:
        row = {"Finding": label}
        for model_name in models:
            row[model_name] = float(per_class_dict.get(model_name, {}).get(label, np.nan))
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, "table3_perfinding.csv")
    txt_path = os.path.join(tables_dir, "table3_perfinding.txt")
    df.to_csv(csv_path, index=False)

    header = f"{'Finding':<24} {'DenseNet-121':>12} {'ProtoPNet':>10} {'ProtoCXR':>10}"
    lines = [
        "=" * len(header),
        "TABLE III - PER-FINDING AUC",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for row in rows:
        dense = row["DenseNet-121"]
        pnet = row["ProtoPNet"]
        pcxr = row["ProtoCXR"]
        dense_str = f"{dense:.3f}" if np.isfinite(dense) else "N/A"
        pnet_str = f"{pnet:.3f}" if np.isfinite(pnet) else "N/A"
        pcxr_str = f"{pcxr:.3f}" if np.isfinite(pcxr) else "N/A"
        lines.append(f"{row['Finding']:<24} {dense_str:>12} {pnet_str:>10} {pcxr_str:>10}")
    lines.append("=" * len(header))

    with open(txt_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")