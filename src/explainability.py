"""
src/explainability.py
=====================
Prototype explanation pipeline and publication-quality visualisations.

Provides:
  - get_prototype_explanation  : Extract explanation dict for one image.
  - visualize_explanation      : 3-panel figure (original / heatmap / prototype).
  - find_nearest_training_patches : Dataset-wide nearest-patch search.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config
from src.model import ProtoCXR

matplotlib.rcParams["font.family"] = "serif"


# ─── Prototype Explanation Extractor ─────────────────────────────────────────

def get_prototype_explanation(
    model: ProtoCXR,
    image_tensor: torch.Tensor,
    class_idx: int,
    device: torch.device,
) -> Dict:
    """Extract an interpretable explanation for a single image–class pair.

    Args:
        model:        Trained :class:`~src.model.ProtoCXR` model.
        image_tensor: Image tensor of shape ``(1, 3, H, W)`` on CPU or device.
        class_idx:    Class index to explain.
        device:       Compute device.

    Returns:
        Dictionary with the following keys:

        - ``proto_idx`` (int): Index of the most-activated prototype.
        - ``class_idx`` (int): The requested class index.
        - ``sim_score`` (float): Peak similarity value for the top prototype.
        - ``spatial_map`` (ndarray, shape ``(7, 7)``): Raw similarity map.
        - ``proto_vector`` (ndarray, shape ``(D,)``): Prototype embedding.
        - ``activation_upsampled`` (ndarray, shape ``(224, 224)``): Bilinearly
          upsampled similarity map for overlay visualisation.
    """
    model.eval()
    x = image_tensor.to(device)

    with torch.no_grad():
        explanation = model.get_explanation(x, class_idx)

    spatial_map = explanation["spatial_map"]           # Tensor (H, W)
    # Bilinear upsample to full image resolution
    up = F.interpolate(
        spatial_map.unsqueeze(0).unsqueeze(0).float(),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return {
        "proto_idx":           explanation["proto_idx"],
        "class_idx":           class_idx,
        "sim_score":           explanation["sim_score"],
        "spatial_map":         spatial_map.cpu().numpy(),
        "proto_vector":        explanation["proto_vector"].cpu().numpy(),
        "activation_upsampled": up,
    }


# ─── Explanation Visualisation ────────────────────────────────────────────────

def visualize_explanation(
    image_np: np.ndarray,
    explanation_dict: Dict,
    class_name: str,
    save_path: Optional[str] = None,
    config: Optional[Config] = None,
) -> plt.Figure:
    """Create a 3-panel publication figure for one prototype explanation.

    Panels:
        1. Original CXR image
        2. Similarity activation heatmap overlaid on image (viridis, α=0.5)
        3. Spatial similarity map thumbnail (7×7 upsampled)

    No title text is placed inside the figure bounds.

    Args:
        image_np:         Image as ``(H, W, 3)`` float or uint8 ndarray.
        explanation_dict: Output of :func:`get_prototype_explanation`.
        class_name:       String label name (used for annotation, not title).
        save_path:        If given, save the figure to this path as PNG.
        config:           ``Config`` for DPI and font settings. Uses defaults
                          if ``None``.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure` object.
    """
    if config is None:
        config = Config()

    _apply_style(config)

    sim_score   = explanation_dict["sim_score"]
    heatmap     = explanation_dict["activation_upsampled"]     # (224, 224)
    spatial_map = explanation_dict["spatial_map"]              # (7, 7)

    # Normalise image to [0, 1] for display
    img_display = image_np.astype(np.float32)
    if img_display.max() > 1.0:
        img_display /= 255.0

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    # Panel 1 — Original image
    axes[0].imshow(img_display, cmap="gray" if img_display.ndim == 2 else None)
    axes[0].axis("off")

    # Panel 2 — Heatmap overlay
    axes[1].imshow(img_display, cmap="gray" if img_display.ndim == 2 else None)
    axes[1].imshow(heatmap, cmap="viridis", alpha=0.5,
                   vmin=heatmap.min(), vmax=heatmap.max())
    axes[1].text(
        0.97, 0.03, f"sim = {sim_score:.2f}",
        transform=axes[1].transAxes,
        ha="right", va="bottom",
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
    )
    axes[1].axis("off")

    # Panel 3 — Prototype spatial map thumbnail
    axes[2].imshow(spatial_map, cmap="viridis", aspect="auto")
    axes[2].axis("off")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.4)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(
            save_path,
            dpi=config.FIG_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
    return fig


# ─── Nearest Training Patches ────────────────────────────────────────────────

def find_nearest_training_patches(
    model: ProtoCXR,
    dataloader: DataLoader,
    device: torch.device,
    n_top: int = 5,
) -> Dict[int, List[Tuple]]:
    """Find the top-N most similar training patches for each prototype.

    Useful for prototype push visualisation and paper figure generation.

    Args:
        model:      :class:`~src.model.ProtoCXR` model (features_only mode).
        dataloader: Training DataLoader (images, labels).
        device:     Compute device.
        n_top:      Number of top patches per prototype.

    Returns:
        Dictionary ``{proto_idx: [(patch_flat_idx, sim_score), ...]}``.
        ``patch_flat_idx`` encodes ``(batch_start_sample_idx, spatial_row, spatial_col)``.
    """
    model.eval()
    total_proto = model.num_classes * model.num_proto

    # Top-N heap: list of (neg_sim, info) for min-heap behaviour
    top_lists: Dict[int, List[Tuple]] = {k: [] for k in range(total_proto)}

    sample_offset = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            feat_list = model.backbone(images)
            features  = model.proj(feat_list[-1])              # (B, D, H, W)
            B, D, H, W = features.shape

            sim_maps = model.sim_fn(features, model.prototypes)  # (B, P, H, W)

            for b in range(B):
                for k in range(total_proto):
                    for i in range(H):
                        for j in range(W):
                            score = sim_maps[b, k, i, j].item()
                            info  = (sample_offset + b, i, j, score)
                            top_lists[k].append(info)

            sample_offset += B

    # For each prototype, keep top-N by sim_score
    for k in range(total_proto):
        top_lists[k] = sorted(top_lists[k], key=lambda x: -x[3])[:n_top]

    return top_lists


# ─── Internal helper ─────────────────────────────────────────────────────────

def _apply_style(config: Config) -> None:
    """Apply global matplotlib style settings from config.

    Args:
        config: ``Config`` instance.
    """
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
