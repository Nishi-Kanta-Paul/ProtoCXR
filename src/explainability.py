"""Explainability helpers for ProtoCXR."""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import ProtoCXR


def get_prototype_explanation(
    model: ProtoCXR,
    image_tensor: torch.Tensor,
    class_idx: int,
    device: torch.device,
) -> Dict[str, object]:
    """Compute a prototype explanation for one image and class.

    Args:
        model: Trained ProtoCXR model.
        image_tensor: Input image tensor shaped (1, 3, 224, 224).
        class_idx: Class index to explain.
        device: Active device.

    Returns:
        Dictionary with proto_idx, class_idx, sim_score, spatial_map,
        activation_upsampled, and proto_vector.

    Raises:
        ValueError: If image_tensor does not have batch size 1.
    """

    if image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must have shape (1, 3, H, W).")

    model.eval()
    with torch.no_grad():
        explanation = model.get_explanation(image_tensor.to(device), class_idx)

    spatial_map = explanation["spatial_map"].detach().cpu().float()
    activation_upsampled = F.interpolate(
        spatial_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    return {
        "proto_idx": int(explanation["proto_idx"]),
        "class_idx": int(class_idx),
        "sim_score": float(explanation["sim_score"]),
        "spatial_map": spatial_map.numpy(),
        "activation_upsampled": activation_upsampled.cpu().numpy(),
        "proto_vector": explanation["proto_vector"].detach().cpu().numpy(),
    }


def visualize_explanation(
    image_np: np.ndarray,
    explanation: Dict[str, object],
    class_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create 3-panel explanation figure.

    Args:
        image_np: Input image array shaped (H, W) or (H, W, 3).
        explanation: Dictionary returned by get_prototype_explanation.
        class_name: Human-readable class name.
        save_path: Optional save path for PNG output.

    Returns:
        Matplotlib figure object.

    Raises:
        ValueError: If prototype vector cannot be reshaped to 16x32.
    """

    del class_name
    image = image_np.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    heatmap = np.asarray(explanation["activation_upsampled"], dtype=np.float32)
    sim_score = float(explanation["sim_score"])
    proto_idx = int(explanation["proto_idx"])
    proto_vector = np.asarray(explanation["proto_vector"], dtype=np.float32)

    if proto_vector.size != 16 * 32:
        raise ValueError("Expected prototype vector dimension 512 for 16x32 reshape.")
    proto_grid = proto_vector.reshape(16, 32)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), facecolor="white")

    # Panel 1: Original image.
    if image.ndim == 2:
        axes[0].imshow(image, cmap="gray")
    else:
        axes[0].imshow(image)
    axes[0].axis("off")

    # Panel 2: Overlayed heatmap.
    if image.ndim == 2:
        axes[1].imshow(image, cmap="gray")
    else:
        axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="viridis", alpha=0.4)
    axes[1].text(
        0.98,
        0.02,
        f"sim = {sim_score:.2f}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
    )
    axes[1].axis("off")

    # Panel 3: Prototype vector visualization.
    axes[2].imshow(proto_grid, cmap="viridis", aspect="auto")
    axes[2].set_xlabel(f"Prototype #{proto_idx}")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    return fig


def find_nearest_patches(
    model: ProtoCXR,
    dataloader: DataLoader,
    device: torch.device,
    n_top: int = 5,
) -> Dict[int, List[Tuple[torch.Tensor, Tuple[int, int], float]]]:
    """Find top-N nearest patches for each prototype.

    Args:
        model: Trained ProtoCXR model.
        dataloader: Dataloader over training images.
        device: Active device.
        n_top: Number of nearest patches per prototype.

    Returns:
        Mapping from prototype index to list of tuples:
        (image_tensor, spatial_position, similarity_score).

    Raises:
        None.
    """

    model.eval()
    result: Dict[int, List[Tuple[torch.Tensor, Tuple[int, int], float]]] = {
        proto_idx: [] for proto_idx in range(model.total_proto)
    }

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            _, sim_maps, _ = model(images, return_sim_maps=True)

            batch_size, _, height, width = sim_maps.shape
            for proto_idx in range(model.total_proto):
                scores = sim_maps[:, proto_idx].reshape(batch_size, height * width)
                flat_scores = scores.reshape(-1)
                k_val = min(n_top, flat_scores.numel())
                top_scores, top_indices = torch.topk(flat_scores, k=k_val)

                for score, flat_idx in zip(top_scores.tolist(), top_indices.tolist()):
                    img_pos = flat_idx // (height * width)
                    spatial = flat_idx % (height * width)
                    row = spatial // width
                    col = spatial % width
                    entry = (images[img_pos].detach().cpu(), (int(row), int(col)), float(score))
                    result[proto_idx].append(entry)

    for proto_idx in result:
        result[proto_idx] = sorted(result[proto_idx], key=lambda item: item[2], reverse=True)[:n_top]

    return result