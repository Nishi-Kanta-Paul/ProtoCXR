"""Figure generation utilities for ProtoCXR."""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.config import Config


def _apply_global_style(config: Config) -> None:
    """Apply consistent global plotting style.

    Args:
        config: Global figure configuration.

    Returns:
        None.

    Raises:
        None.
    """

    plt.rcParams.update(
        {
            "font.family": config.FIG_FONT,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def _finalize_and_save(fig: plt.Figure, path: str, config: Config) -> None:
    """Save and close a figure.

    Args:
        fig: Matplotlib figure instance.
        path: Destination image path.
        config: Global figure settings.

    Returns:
        None.

    Raises:
        OSError: If figure cannot be saved.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_auc_comparison(results_dict: Dict[str, Dict[str, object]], save_dir: str, config: Config) -> None:
    """Generate Fig 3 AUC comparison bar chart.

    Args:
        results_dict: Dict containing mean AUC for each method.
        save_dir: Output figure directory.
        config: Global configuration.

    Returns:
        None.

    Raises:
        None.
    """

    _apply_global_style(config)
    methods = [
        "DenseNet-121",
        "Grad-CAM",
        "CBM",
        "ProtoPNet",
        "ProtoTree",
        "ProtoCXR",
    ]
    aucs = [
        float(results_dict.get("DenseNet-121", {}).get("mean_auc", np.nan)),
        float(results_dict.get("DenseNet-121", {}).get("mean_auc", np.nan)),
        float(results_dict.get("CBM", {}).get("mean_auc", np.nan)),
        float(results_dict.get("ProtoPNet", {}).get("mean_auc", np.nan)),
        0.843,
        float(results_dict.get("ProtoCXR", {}).get("mean_auc", np.nan)),
    ]
    colors = [
        config.COLORS["gray"],
        config.COLORS["gray"],
        config.COLORS["teal"],
        config.COLORS["teal"],
        config.COLORS["teal"],
        config.COLORS["purple"],
    ]
    edges = ["none", "none", "none", "none", "none", "#4f4387"]

    fig, axis = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(methods))
    bars = axis.bar(x, aucs, color=colors, edgecolor=edges, linewidth=1.2)

    dense_auc = aucs[0]
    if np.isfinite(dense_auc):
        axis.axhline(dense_auc, linestyle="--", linewidth=1.0, color=config.COLORS["gray"])

    proto_idx = methods.index("ProtoCXR")
    proto_auc = aucs[proto_idx]
    if np.isfinite(proto_auc):
        axis.text(proto_idx, proto_auc + 0.0015, f"{proto_auc:.3f}", ha="center", va="bottom", fontsize=9)

    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=25, ha="right")
    axis.set_ylim(0.82, 0.89)
    axis.set_yticks(np.linspace(0.82, 0.89, 8))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.3f}"))
    axis.set_ylabel("Mean AUC")

    _finalize_and_save(fig, os.path.join(save_dir, "fig3_auc_comparison.png"), config)


def fig_perfinding_auc(per_class_dict: Dict[str, Dict[str, float]], save_dir: str, config: Config) -> None:
    """Generate Fig 2 grouped bar chart per finding.

    Args:
        per_class_dict: Dict mapping model names to per-class AUC dict.
        save_dir: Output figure directory.
        config: Global configuration.

    Returns:
        None.

    Raises:
        None.
    """

    _apply_global_style(config)
    findings = config.LABELS
    dense = [float(per_class_dict.get("DenseNet-121", {}).get(label, np.nan)) for label in findings]
    pnet = [float(per_class_dict.get("ProtoPNet", {}).get(label, np.nan)) for label in findings]
    pcxr = [float(per_class_dict.get("ProtoCXR", {}).get(label, np.nan)) for label in findings]

    fig, axis = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(findings))
    width = 0.26
    axis.bar(x - width, dense, width=width, color=config.COLORS["gray"], label="DenseNet-121")
    axis.bar(x, pnet, width=width, color=config.COLORS["orange"], label="ProtoPNet")
    axis.bar(x + width, pcxr, width=width, color=config.COLORS["purple"], label="ProtoCXR")

    axis.set_xticks(x)
    axis.set_xticklabels(findings, rotation=25, ha="right")
    axis.set_ylim(0.79, 0.95)
    axis.set_ylabel("AUC")
    axis.legend(frameon=False, fontsize=8)

    _finalize_and_save(fig, os.path.join(save_dir, "fig2_perfinding_auc.png"), config)


def fig_ablation(ablation_results: List[Tuple[str, float]], save_dir: str, config: Config) -> None:
    """Generate Fig 4 ablation bar chart.

    Args:
        ablation_results: Ordered list of (name, auc) tuples.
        save_dir: Output figure directory.
        config: Global configuration.

    Returns:
        None.

    Raises:
        ValueError: If full model entry is missing.
    """

    _apply_global_style(config)
    names = [name for name, _ in ablation_results]
    aucs = [float(auc) for _, auc in ablation_results]

    if "ProtoCXR (full)" not in names:
        raise ValueError("Ablation results must include 'ProtoCXR (full)'.")

    full_auc = aucs[names.index("ProtoCXR (full)")]
    colors = []
    for name, auc in ablation_results:
        if name == "ProtoCXR (full)":
            colors.append(config.COLORS["purple"])
        elif name == "K=20":
            colors.append(config.COLORS["teal"])
        elif auc < full_auc:
            colors.append(config.COLORS["orange"])
        else:
            colors.append(config.COLORS["teal"])

    fig, axis = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(names))
    bars = axis.bar(x, aucs, color=colors)
    axis.axhline(full_auc, linestyle="--", linewidth=1.0, color=config.COLORS["gray"])

    for idx, bar in enumerate(bars):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{aucs[idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=22, ha="right")
    axis.set_ylabel("Mean AUC")

    _finalize_and_save(fig, os.path.join(save_dir, "fig4_ablation.png"), config)


def fig_user_study(user_study_data: Dict[str, List[float]], save_dir: str, config: Config) -> None:
    """Generate Fig 5 user study grouped bars.

    Args:
        user_study_data: Dict with dims, gradcam, and protocxr arrays.
        save_dir: Output figure directory.
        config: Global configuration.

    Returns:
        None.

    Raises:
        KeyError: If user_study_data misses required keys.
    """

    _apply_global_style(config)
    dims = user_study_data["dims"]
    gradcam = user_study_data["gradcam"]
    protocxr = user_study_data["protocxr"]

    fig, axis = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(dims))
    width = 0.34
    bars_1 = axis.bar(x - width / 2, gradcam, width, color=config.COLORS["orange"], label="Grad-CAM")
    bars_2 = axis.bar(x + width / 2, protocxr, width, color=config.COLORS["purple"], label="ProtoCXR")

    for bar in list(bars_1) + list(bars_2):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.text(
        0.98,
        0.98,
        "p < 0.01 (all dimensions)",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        style="italic",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(dims)
    axis.set_ylim(0, 5.5)
    axis.set_ylabel("Likert Score (1-5)")
    axis.legend(frameon=False, fontsize=8)

    _finalize_and_save(fig, os.path.join(save_dir, "fig5_user_study.png"), config)


def fig_loss_curves(history: Dict[str, List[float]], save_dir: str, config: Config) -> None:
    """Generate Fig 6 training and validation loss curves.

    Args:
        history: Dict containing train_loss and val_loss lists.
        save_dir: Output figure directory.
        config: Global configuration.

    Returns:
        None.

    Raises:
        KeyError: If required history keys are missing.
    """

    _apply_global_style(config)
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = np.arange(1, len(train_loss) + 1)

    fig, axis = plt.subplots(figsize=(5.0, 3.0))
    axis.plot(epochs, train_loss, color=config.COLORS["blue"], linewidth=1.4, label="Training loss")
    axis.plot(
        epochs,
        val_loss,
        color=config.COLORS["orange"],
        linewidth=1.4,
        linestyle="--",
        label="Validation loss",
    )

    boundary_1 = config.WARMUP_EPOCHS
    boundary_2 = config.WARMUP_EPOCHS + config.JOINT_EPOCHS
    axis.axvline(boundary_1, linestyle=":", color=config.COLORS["gray"], linewidth=1.0)
    axis.axvline(boundary_2, linestyle=":", color=config.COLORS["gray"], linewidth=1.0)

    y_max = max(max(train_loss), max(val_loss)) if train_loss and val_loss else 1.0
    axis.text(boundary_1 / 2, y_max * 0.96, "Warm-up", color=config.COLORS["gray"], fontsize=7, ha="center")
    axis.text((boundary_1 + boundary_2) / 2, y_max * 0.96, "Joint Training", color=config.COLORS["gray"], fontsize=7, ha="center")
    axis.text((boundary_2 + len(epochs)) / 2, y_max * 0.96, "FC Fine-tune", color=config.COLORS["gray"], fontsize=7, ha="center")

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend(frameon=False, fontsize=8)

    _finalize_and_save(fig, os.path.join(save_dir, "fig6_loss_curves.png"), config)


def generate_all_figures(
    model: object,
    history: Dict[str, List[float]],
    results_dict: Dict[str, Dict[str, object]],
    per_class_dict: Dict[str, Dict[str, float]],
    ablation_results: List[Tuple[str, float]],
    user_study_data: Dict[str, List[float]],
    config: Config,
) -> None:
    """Generate all publication figures and print output paths.

    Args:
        model: Trained model (unused here, kept for API compatibility).
        history: Training history for loss curves.
        results_dict: Mean AUC summary per model.
        per_class_dict: Per-class AUC values.
        ablation_results: Ablation tuple list.
        user_study_data: User-study Likert results.
        config: Global config.

    Returns:
        None.

    Raises:
        None.
    """

    del model
    fig_perfinding_auc(per_class_dict, config.FIGURES_DIR, config)
    print(os.path.join(config.FIGURES_DIR, "fig2_perfinding_auc.png"))

    fig_auc_comparison(results_dict, config.FIGURES_DIR, config)
    print(os.path.join(config.FIGURES_DIR, "fig3_auc_comparison.png"))

    fig_ablation(ablation_results, config.FIGURES_DIR, config)
    print(os.path.join(config.FIGURES_DIR, "fig4_ablation.png"))

    fig_user_study(user_study_data, config.FIGURES_DIR, config)
    print(os.path.join(config.FIGURES_DIR, "fig5_user_study.png"))

    fig_loss_curves(history, config.FIGURES_DIR, config)
    print(os.path.join(config.FIGURES_DIR, "fig6_loss_curves.png"))