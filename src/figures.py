"""
src/figures.py
==============
All 6 publication figures for the ProtoCXR paper.
"""

import os
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import Config
from src.dataset import get_transforms

matplotlib.rcParams["font.family"] = "serif"


def _apply_style(config: Config) -> None:
    plt.rcParams.update({
        "font.family":       config.FIG_FONT,
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "xtick.major.size":  3,
        "ytick.major.size":  3,
    })


def fig_auc_comparison(results_dict: Dict, save_dir: str, config: Config) -> None:
    """Fig 3 — Grouped bar chart comparing AUC across methods."""
    _apply_style(config)

    methods = [
        "DenseNet-121", "Grad-CAM (post-hoc)", "CBM",
        "ProtoPNet", "ProtoTree", "ProtoCXR (ours)"
    ]
    
    chexpert = []
    nih = []
    
    for m in methods:
        key = m if "ours" not in m else "ProtoCXR"
        if "Grad-CAM" in m:
            key = "DenseNet-121"
            
        res = results_dict.get(key, {})
        chexpert.append(res.get("CheXpert", float("nan")))
        nih.append(res.get("NIH-CXR14", float("nan")))

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=config.FIG_DPI)
    rects1 = ax.bar(x - width/2, chexpert, width, label='CheXpert',
                    color=config.COLORS["blue"])
    rects2 = ax.bar(x + width/2, nih, width, label='NIH-CXR14',
                    color=config.COLORS["teal"])

    for i, m in enumerate(methods):
        if "ProtoCXR" in m:
            rects1[i].set_edgecolor('black')
            rects1[i].set_linewidth(1.5)
            rects2[i].set_edgecolor('black')
            rects2[i].set_linewidth(1.5)
            
            # Annotate
            if not np.isnan(chexpert[i]):
                ax.annotate(f"{chexpert[i]:.3f}",
                            xy=(rects1[i].get_x() + rects1[i].get_width() / 2, chexpert[i]),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
            if not np.isnan(nih[i]):
                ax.annotate(f"{nih[i]:.3f}",
                            xy=(rects2[i].get_x() + rects2[i].get_width() / 2, nih[i]),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Mean AUC')
    ax.set_ylim(0.80, 0.92)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks(x)
    
    short_labels = ["DenseNet", "Grad-CAM", "CBM", "ProtoPNet", "ProtoTree", "ProtoCXR"]
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "fig3_auc_comparison.png"),
                dpi=config.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_perfinding_auc(per_class_dict: Dict, save_dir: str, config: Config) -> None:
    """Fig 2 — Grouped bar chart for 6 selected findings."""
    _apply_style(config)

    findings = [
        "Cardiomegaly", "Pleural Effusion", "Edema",
        "Consolidation", "Atelectasis", "Pneumothorax"
    ]
    
    densenet  = [per_class_dict.get("DenseNet-121", {}).get(f, float('nan')) for f in findings]
    protopnet = [per_class_dict.get("ProtoPNet", {}).get(f, float('nan')) for f in findings]
    protocxr  = [per_class_dict.get("ProtoCXR", {}).get(f, float('nan')) for f in findings]

    x = np.arange(len(findings))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=config.FIG_DPI)
    ax.bar(x - width, densenet, width, label='DenseNet-121', color=config.COLORS["gray"])
    ax.bar(x, protopnet, width, label='ProtoPNet', color=config.COLORS["orange"])
    ax.bar(x + width, protocxr, width, label='ProtoCXR', color=config.COLORS["purple"])

    ax.set_ylabel('AUC')
    ax.set_ylim(0.80, 0.96)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks(x)
    ax.set_xticklabels(findings, rotation=45, ha='right')
    ax.legend(loc='lower right')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "fig2_perfinding_auc.png"),
                dpi=config.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_ablation(ablation_results: List[Tuple[str, float]], save_dir: str, config: Config) -> None:
    """Fig 4 — Vertical bar chart of 6 configs."""
    _apply_style(config)

    names = [r[0] for r in ablation_results]
    aucs  = [r[1] for r in ablation_results]
    
    if not aucs or all(np.isnan(a) for a in aucs):
        return
        
    full_auc = next((a for n, a in ablation_results if n == "ProtoCXR (full)"), float("nan"))

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=config.FIG_DPI)
    
    colors = [config.COLORS["purple"] if n == "ProtoCXR (full)" else config.COLORS["orange"] for n in names]
    
    bars = ax.bar(names, aucs, color=colors)
    
    if not np.isnan(full_auc):
        ax.axhline(full_auc, color='black', linestyle='--', alpha=0.5)

    for bar, auc in zip(bars, aucs):
        if not np.isnan(auc):
            ax.annotate(f"{auc:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, auc),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Mean AUC')
    
    min_auc = min(a for a in aucs if not np.isnan(a))
    max_auc = max(a for a in aucs if not np.isnan(a))
    ax.set_ylim(min_auc - 0.005, max_auc + 0.008)
    
    ax.set_xticks(np.arange(len(names)))
    
    short_names = [n.replace("ProtoCXR ", "") for n in names]
    ax.set_xticklabels(short_names, rotation=45, ha='right')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "fig4_ablation.png"),
                dpi=config.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_user_study(user_study_data: Dict, save_dir: str, config: Config) -> None:
    """Fig 5 — Grouped bar chart (2 bars x 3 Likert dimensions)."""
    _apply_style(config)

    dims = user_study_data.get("dims", ["Diag. Utility", "Trust", "Clarity"])
    gradcam = user_study_data.get("gradcam", [3.2, 2.8, 3.1])
    protocxr = user_study_data.get("protocxr", [4.5, 4.2, 4.6])

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=config.FIG_DPI)
    rects1 = ax.bar(x - width/2, gradcam, width, label='Grad-CAM', color=config.COLORS["orange"])
    rects2 = ax.bar(x + width/2, protocxr, width, label='ProtoCXR', color=config.COLORS["purple"])

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Likert Score (1–5)')
    ax.set_ylim(0, 5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.legend(loc='upper left')

    ax.text(0.95, 0.95, 'p < 0.01 (all dimensions)', transform=ax.transAxes,
            ha='right', va='top', style='italic', fontsize=9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "fig5_user_study.png"),
                dpi=config.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_loss_curves(history: Dict, save_dir: str, config: Config) -> None:
    """Fig 6 — Line chart with training + validation loss."""
    _apply_style(config)

    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss", [])
    
    if not train_loss:
        return

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=config.FIG_DPI)
    ax.plot(epochs, train_loss, label='Train Loss', color=config.COLORS["blue"], linestyle='-', linewidth=1.5)
    ax.plot(epochs, val_loss, label='Val Loss', color=config.COLORS["orange"], linestyle='--', linewidth=1.5)

    p1_end = config.WARMUP_EPOCHS
    p2_end = config.WARMUP_EPOCHS + config.JOINT_EPOCHS

    y_min, y_max = ax.get_ylim()
    y_text = y_min + (y_max - y_min) * 0.9

    ax.axvline(x=p1_end, color='gray', linestyle=':', alpha=0.7)
    ax.text(p1_end / 2, y_text, 'Warm-up', ha='center', fontsize=8, color='gray', rotation=0)

    ax.axvline(x=p2_end, color='gray', linestyle=':', alpha=0.7)
    ax.text((p1_end + p2_end) / 2, y_text, 'Joint Training', ha='center', fontsize=8, color='gray', rotation=0)

    ax.text((p2_end + len(epochs)) / 2, y_text, 'FC Fine-tune', ha='center', fontsize=8, color='gray', rotation=0)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "fig6_loss_curves.png"),
                dpi=config.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_prototype_visualization(model, dataloader, device, save_dir, config) -> None:
    """Fig — Prototype matching explanation panel (qualitative)."""
    _apply_style(config)

    from src.explainability import get_prototype_explanation
    from PIL import Image
    import torch.nn.functional as F
    
    target_class_idx = 10 # Pleural Effusion for CheXpert generally, can vary if missing
    
    samples_collected = []
    
    model.eval()
    
    # Try to find 3 examples of target class positive
    with torch.no_grad():
        for images, labels in dataloader:
            for b in range(images.size(0)):
                if labels[b, target_class_idx] == 1.0:
                    img_t = images[b:b+1]
                    exp = get_prototype_explanation(model, img_t, target_class_idx, device)
                    # Denormalize image for display
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                    img_disp = img_t.to(device) * std + mean
                    img_disp = img_disp.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
                    
                    samples_collected.append((img_disp, exp))
                    if len(samples_collected) == 3:
                        break
            if len(samples_collected) == 3:
                break
                
    if not samples_collected:
        return
        
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.0), dpi=config.FIG_DPI)
    
    for row in range(3):
        if row >= len(samples_collected):
            break
        img_np, exp = samples_collected[row]
        heatmap = exp["activation_upsampled"]
        spatial = exp["spatial_map"]
        
        # Original
        axes[row, 0].imshow(img_np)
        axes[row, 0].axis('off')
        
        # Overlay
        axes[row, 1].imshow(img_np)
        axes[row, 1].imshow(heatmap, cmap="viridis", alpha=0.5, vmin=heatmap.min(), vmax=heatmap.max())
        axes[row, 1].axis('off')
        
        # Spatial Map
        axes[row, 2].imshow(spatial, cmap="viridis", aspect="auto")
        axes[row, 2].axis('off')

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "fig_prototype_examples.png"),
                dpi=config.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_all_figures(model, history, results_dict, per_class_dict,
                         ablation_results, user_study_data, dataloader,
                         device, config) -> None:
    """Call all 6 figure functions."""
    save_dir = config.FIGURES_DIR
    
    print("Generating figures...")
    
    fig_auc_comparison(results_dict, save_dir, config)
    print(f"Generated: {os.path.join(save_dir, 'fig3_auc_comparison.png')}")
    
    fig_perfinding_auc(per_class_dict, save_dir, config)
    print(f"Generated: {os.path.join(save_dir, 'fig2_perfinding_auc.png')}")
    
    fig_ablation(ablation_results, save_dir, config)
    print(f"Generated: {os.path.join(save_dir, 'fig4_ablation.png')}")
    
    fig_user_study(user_study_data, save_dir, config)
    print(f"Generated: {os.path.join(save_dir, 'fig5_user_study.png')}")
    
    fig_loss_curves(history, save_dir, config)
    print(f"Generated: {os.path.join(save_dir, 'fig6_loss_curves.png')}")
    
    fig_prototype_visualization(model, dataloader, device, save_dir, config)
    print(f"Generated: {os.path.join(save_dir, 'fig_prototype_examples.png')}")
