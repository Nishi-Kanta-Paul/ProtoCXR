"""
src/model.py
============
ProtoCXR model and all submodules:
  - PrototypeSimilarity
  - LungMaskNet
  - ProtoCXR (main model)
"""

import warnings
from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config


# ─── Prototype Similarity ─────────────────────────────────────────────────────

class PrototypeSimilarity(nn.Module):
    """Compute log-similarity between spatial feature maps and prototypes.

    Implements:
        g(z, p) = log((||z - p||² + 1) / (||z - p||² + epsilon))

    Args:
        epsilon: Small constant for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-prototype similarity maps.

        Args:
            features:   ``(B, D, H, W)`` spatial feature maps.
            prototypes: ``(P, D)`` prototype vectors.

        Returns:
            Similarity maps of shape ``(B, P, H, W)``.
        """
        B, D, H, W = features.shape
        P = prototypes.shape[0]

        # Reshape features → (B, H*W, D)
        f_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Reshape prototypes → (1, P, D)
        p = prototypes.unsqueeze(0)  # (1, P, D)

        # ||z - p||² via broadcasting: (B, H*W, 1, D) - (1, 1, P, D)
        f_exp = f_flat.unsqueeze(2)  # (B, H*W, 1, D)
        p_exp = p.unsqueeze(0)       # (1, 1, P, D)

        # Squared L2 distances: (B, H*W, P)
        dist_sq = ((f_exp - p_exp) ** 2).sum(dim=-1)

        # Log similarity: log((d² + 1) / (d² + ε))
        sim = torch.log((dist_sq + 1.0) / (dist_sq + self.epsilon))

        # Reshape → (B, P, H, W)
        sim = sim.permute(0, 2, 1).reshape(B, P, H, W)
        return sim


# ─── Lung Mask Network ────────────────────────────────────────────────────────

class LungMaskNet(nn.Module):
    """Lightweight U-Net for lung region segmentation.

    Produces a binary (0/1) mask at 7×7 spatial resolution for use in
    the Anatomical Region Alignment (ARA) loss.

    Note:
        In production, load pretrained weights from:
        https://github.com/imlab-uiip/lung-segmentation-2d
        If weights are not available, random initialization is used with
        a printed warning.

    Args:
        pretrained_path: Optional path to saved weight file (``.pth``).
        threshold: Binarization threshold for the sigmoid output.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.threshold = threshold

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        if pretrained_path is not None:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                self.load_state_dict(state, strict=False)
                print(f"[LungMaskNet] Loaded pretrained weights from: {pretrained_path}")
            except Exception as exc:
                warnings.warn(
                    f"[LungMaskNet] Could not load weights from '{pretrained_path}': {exc}. "
                    "Using random initialization.",
                    RuntimeWarning,
                )
        else:
            print(
                "[LungMaskNet] No pretrained weights provided.\n"
                "  → For best results, download pretrained lung segmentation weights from:\n"
                "    https://github.com/imlab-uiip/lung-segmentation-2d\n"
                "  → Continuing with random initialization (ARA loss will still work but\n"
                "    may not perfectly align prototypes to lung anatomy during warm-up)."
            )

        # Freeze all parameters by default
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise lung probability map.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.

        Returns:
            Sigmoid probability map of shape ``(B, 1, H/2, W/2)``.
        """
        enc = self.enc1(x)
        pooled = self.pool(enc)
        bottleneck = self.bottleneck(pooled)
        # Upsample back to input size
        up = F.interpolate(bottleneck, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = self.dec1(up)
        return out

    @torch.no_grad()
    def get_7x7_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Compute binary lung mask at 7×7 spatial resolution.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.

        Returns:
            Binary mask tensor of shape ``(B, 1, 7, 7)`` with values in ``{0, 1}``.
        """
        prob_map = self.forward(x)
        # Downsample to 7×7
        mask_7 = F.interpolate(prob_map, size=(7, 7), mode="bilinear", align_corners=False)
        return (mask_7 >= self.threshold).float()


# ─── ProtoCXR Main Model ──────────────────────────────────────────────────────

class ProtoCXR(nn.Module):
    """Prototype-Based Interpretable Multi-Label CXR Classifier.

    Architecture:
        Input (B, 3, 224, 224)
            → DenseNet-121 backbone (spatial feature maps, truncated)
            → 1×1 Conv projection (1024 → FEAT_DIM)
            → PrototypeSimilarity layer (FEAT_DIM → C*K similarity maps)
            → Global max-pool over (H, W) per prototype → (B, C*K)
            → Non-negative FC (C*K → C) → logits
            → Sigmoid → predictions

    Args:
        num_classes: Number of diagnostic classes C (default 14).
        num_proto: Number of prototypes per class K.
        feat_dim: Prototype and projection dimension D.
        backbone_name: ``timm`` model identifier for the backbone.
        backbone_pretrained: Load ImageNet-pretrained backbone weights.
        sim_epsilon: Epsilon for :class:`PrototypeSimilarity`.
        pretrained_mask_path: Optional path to ``LungMaskNet`` weights.
    """

    def __init__(
        self,
        num_classes: int,
        num_proto: int,
        feat_dim: int,
        backbone_name: str = "densenet121",
        backbone_pretrained: bool = True,
        sim_epsilon: float = 1e-4,
        pretrained_mask_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_proto = num_proto
        self.feat_dim = feat_dim
        total_proto = num_classes * num_proto

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=backbone_pretrained,
            features_only=True,
        )
        # Infer backbone output channels from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            backbone_out_ch = feats[-1].shape[1]

        # ── Projection 1×1 Conv ───────────────────────────────────────────────
        self.proj = nn.Conv2d(backbone_out_ch, feat_dim, kernel_size=1, bias=False)

        # ── Prototype layer ───────────────────────────────────────────────────
        self.prototypes = nn.Parameter(torch.empty(total_proto, feat_dim))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0)).squeeze_(0)

        # ── Similarity function ───────────────────────────────────────────────
        self.sim_fn = PrototypeSimilarity(epsilon=sim_epsilon)

        # ── Non-negative FC ───────────────────────────────────────────────────
        self.fc = nn.Linear(total_proto, num_classes, bias=False)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc.weight)
            self.fc.weight.abs_()

        # ── Prototype → class mapping ─────────────────────────────────────────
        proto_class_map = torch.arange(num_classes).repeat_interleave(num_proto)
        self.register_buffer("proto_class_map", proto_class_map)

        # ── Lung mask network (frozen) ────────────────────────────────────────
        self.lung_net = LungMaskNet(pretrained_path=pretrained_mask_path)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_sim_maps: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Run a forward pass through ProtoCXR.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.
            return_sim_maps: If ``True``, also return raw similarity maps
                             and projected feature maps.

        Returns:
            If ``return_sim_maps=False``:
                ``(logits, proto_activations)`` where:
                - ``logits``: ``(B, C)``
                - ``proto_activations``: ``(B, C*K)``
            If ``return_sim_maps=True``:
                ``(logits, sim_maps, features)`` where:
                - ``logits``: ``(B, C)``
                - ``sim_maps``: ``(B, C*K, H, W)``
                - ``features``: ``(B, D, H, W)``
        """
        # Enforce non-negative FC weights at every forward pass
        with torch.no_grad():
            self.fc.weight.clamp_(min=0.0)

        # Feature extraction (take last feature level)
        feat_list = self.backbone(x)
        features = self.proj(feat_list[-1])          # (B, D, H, W)

        # Prototype similarity maps
        sim_maps = self.sim_fn(features, self.prototypes)  # (B, C*K, H, W)

        # Global max-pool over spatial dims
        proto_activations = sim_maps.amax(dim=(2, 3))      # (B, C*K)

        # Class logits via non-negative FC
        logits = self.fc(proto_activations)                 # (B, C)

        if return_sim_maps:
            return logits, sim_maps, features
        return logits, proto_activations

    # ── Explanation ───────────────────────────────────────────────────────────

    def get_explanation(
        self,
        x: torch.Tensor,
        class_idx: int,
    ) -> Dict:
        """Return an interpretable explanation for a single image–class pair.

        Args:
            x: Single image tensor of shape ``(1, 3, H, W)``.
            class_idx: Index of the class to explain.

        Returns:
            Dictionary with keys:
                - ``proto_idx`` (int): Most-activated prototype index.
                - ``spatial_map`` (Tensor[7,7]): Similarity activation map.
                - ``sim_score`` (float): Maximum similarity score.
                - ``proto_vector`` (Tensor[D]): Prototype embedding vector.

        Raises:
            ValueError: If ``x`` does not have batch size 1.
        """
        if x.shape[0] != 1:
            raise ValueError("get_explanation expects a single image (batch size 1).")

        self.eval()
        with torch.no_grad():
            _, sim_maps, _ = self.forward(x, return_sim_maps=True)
            # sim_maps: (1, C*K, H, W)

            # Select prototypes belonging to the requested class
            class_mask = self.proto_class_map == class_idx       # (C*K,)
            class_proto_indices = class_mask.nonzero(as_tuple=True)[0]

            # Per-prototype max activations for this image
            maps_for_class = sim_maps[0, class_proto_indices]    # (K, H, W)
            max_acts = maps_for_class.amax(dim=(1, 2))           # (K,)

            best_local = max_acts.argmax().item()
            best_proto_idx = class_proto_indices[best_local].item()

            spatial_map = sim_maps[0, best_proto_idx]            # (H, W)
            sim_score   = spatial_map.max().item()
            proto_vec   = self.prototypes[best_proto_idx]        # (D,)

        return {
            "proto_idx":   int(best_proto_idx),
            "spatial_map": spatial_map,
            "sim_score":   float(sim_score),
            "proto_vector": proto_vec,
        }

    # ── Prototype Push ────────────────────────────────────────────────────────

    @torch.no_grad()
    def push_prototypes(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        """Replace prototype vectors with nearest training patch embeddings.

        Phase 3 — Prototype Push Algorithm:
            1. Iterate the full training DataLoader with no gradients.
            2. Extract projected feature maps for each batch.
            3. For each prototype p_k, find the patch z* across all spatial
               locations and images that minimises ||z* - p_k||².
            4. Replace ``self.prototypes[k] ← z*``.

        Logs the mean displacement (L2 change) after the push.

        Args:
            dataloader: Training DataLoader (images, labels).
            device: Compute device.
        """
        self.eval()
        total_proto = self.num_classes * self.num_proto

        best_dist  = torch.full((total_proto,), float("inf"), device=device)
        best_patch = torch.zeros_like(self.prototypes)

        for images, _ in dataloader:
            images = images.to(device)

            feat_list = self.backbone(images)
            features  = self.proj(feat_list[-1])      # (B, D, H, W)

            B, D, H, W = features.shape
            # Flatten spatial: (B * H * W, D)
            patches = features.permute(0, 2, 3, 1).reshape(-1, D)
            # (B*H*W, 1, D) - (1, P, D) → (B*H*W, P) squared distances
            diff     = patches.unsqueeze(1) - self.prototypes.unsqueeze(0)
            dist_sq  = (diff ** 2).sum(dim=-1)        # (B*H*W, P)
            min_dists, min_ids = dist_sq.min(dim=0)   # (P,) each

            improve_mask = min_dists < best_dist
            for k in improve_mask.nonzero(as_tuple=True)[0]:
                k = k.item()
                best_dist[k]  = min_dists[k]
                best_patch[k] = patches[min_ids[k]]

        old_protos = self.prototypes.data.clone()
        self.prototypes.data.copy_(best_patch)

        mean_displacement = (self.prototypes.data - old_protos).norm(dim=1).mean().item()
        print(f"Prototype push complete. Mean displacement: {mean_displacement:.4f}")
        self.train()

    # ── Freeze helpers ────────────────────────────────────────────────────────

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters.

        Args:
            freeze: If ``True``, disables gradients for the backbone
                    (including projection layer). If ``False``, enables them.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        for param in self.proj.parameters():
            param.requires_grad = not freeze

    def freeze_prototypes(self, freeze: bool = True) -> None:
        """Freeze or unfreeze prototype parameters.

        Args:
            freeze: If ``True``, disables gradient updates for
                    ``self.prototypes``. If ``False``, enables them.
        """
        self.prototypes.requires_grad = not freeze
