"""Model definitions for ProtoCXR."""

import os
import warnings
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config


class PrototypeSimilarity(nn.Module):
    """Computes prototype similarity maps from spatial features.

    Args:
        epsilon: Numerical stability constant for denominator.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute similarity map g(z, p) at every spatial location.

        Args:
            features: Tensor shaped (B, D, H, W).
            prototypes: Tensor shaped (P, D).

        Returns:
            Similarity tensor shaped (B, P, H, W).

        Raises:
            ValueError: If feature/prototype dimensions do not match.
        """

        if features.dim() != 4 or prototypes.dim() != 2:
            raise ValueError("features must be (B,D,H,W) and prototypes must be (P,D).")
        if features.shape[1] != prototypes.shape[1]:
            raise ValueError("Feature channels and prototype dimensions must match.")

        bsz, dim, height, width = features.shape
        num_proto = prototypes.shape[0]

        patches = features.permute(0, 2, 3, 1).reshape(bsz, height * width, dim)
        dist_sq = (patches.unsqueeze(2) - prototypes.unsqueeze(0).unsqueeze(0)).pow(2).sum(dim=-1)
        sim = torch.log((dist_sq + 1.0) / (dist_sq + self.epsilon))
        sim = sim.permute(0, 2, 1).reshape(bsz, num_proto, height, width)
        return sim


class LungMaskNet(nn.Module):
    """Frozen lightweight U-Net that outputs binary 7x7 lung masks.

    Args:
        config: Global project config for loading pretrained lung weights.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        weight_path = os.path.join(config.DRIVE_ROOT, "lung_unet.pth")
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state, strict=False)
        else:
            warnings.warn(
                f"LungMaskNet weights not found at {weight_path}. Using random initialization.",
                RuntimeWarning,
            )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run lung mask network forward pass.

        Args:
            x: Input image batch shaped (B, 3, H, W).

        Returns:
            Dense probability masks shaped (B, 1, H, W).

        Raises:
            None.
        """

        enc = self.encoder(x)
        pooled = self.pool(enc)
        bridge = self.bridge(pooled)
        upsampled = F.interpolate(bridge, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.decoder(upsampled)

    @torch.no_grad()
    def get_7x7_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return no-grad binary masks at 7x7 resolution.

        Args:
            x: Input image batch shaped (B, 3, 224, 224).

        Returns:
            Binary lung mask tensor shaped (B, 1, 7, 7).

        Raises:
            None.
        """

        dense = self.forward(x)
        reduced = F.interpolate(dense, size=(7, 7), mode="bilinear", align_corners=False)
        return (reduced >= 0.5).float()


class ProtoCXR(nn.Module):
    """Prototype-based interpretable multi-label classifier for CXR.

    Args:
        config: Global project config.

    Returns:
        None.

    Raises:
        ValueError: If the backbone does not produce expected channels.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.num_proto = config.NUM_PROTO
        self.total_proto = self.num_classes * self.num_proto

        self.backbone = timm.create_model(
            config.BACKBONE,
            features_only=True,
            pretrained=config.BACKBONE_PRETRAINED,
        )

        # Spec requires 1024 -> FEAT_DIM projection for DenseNet-121 features.
        self.proj = nn.Conv2d(1024, config.FEAT_DIM, kernel_size=1)

        self.prototypes = nn.Parameter(
            torch.empty(self.total_proto, config.FEAT_DIM)
        )
        nn.init.xavier_uniform_(self.prototypes)

        self.fc = nn.Linear(self.total_proto, self.num_classes, bias=False)
        nn.init.kaiming_uniform_(self.fc.weight, a=5 ** 0.5)
        with torch.no_grad():
            self.fc.weight.copy_(self.fc.weight.abs())

        self.register_buffer(
            "proto_class_map",
            torch.arange(self.num_classes, dtype=torch.long).repeat_interleave(self.num_proto),
        )

        self.sim_fn = PrototypeSimilarity(config.SIM_EPSILON)
        self.lung_net = LungMaskNet(config)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract projected spatial embeddings.

        Args:
            x: Input image batch shaped (B, 3, H, W).

        Returns:
            Projected feature maps shaped (B, FEAT_DIM, 7, 7).

        Raises:
            ValueError: If DenseNet output channel count is unexpected.
        """

        feats = self.backbone(x)[-1]
        if feats.shape[1] != 1024:
            raise ValueError(
                f"Expected DenseNet last feature channels=1024, got {feats.shape[1]}."
            )
        return self.proj(feats)

    def forward(
        self,
        x: torch.Tensor,
        return_sim_maps: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Run forward pass through ProtoCXR.

        Args:
            x: Input image batch shaped (B, 3, 224, 224).
            return_sim_maps: Whether to return full similarity maps.

        Returns:
            If return_sim_maps is True: (logits, sim_maps, feats).
            Else: (logits, proto_acts).

        Raises:
            None.
        """

        with torch.no_grad():
            self.fc.weight.clamp_(min=0.0)

        feats = self._extract_features(x)
        sim_maps = self.sim_fn(feats, self.prototypes)
        proto_acts = sim_maps.amax(dim=(2, 3))
        logits = self.fc(proto_acts)

        if return_sim_maps:
            return logits, sim_maps, feats
        return logits, proto_acts

    @torch.no_grad()
    def get_explanation(self, x: torch.Tensor, class_idx: int) -> Dict[str, object]:
        """Get the most activated prototype explanation for one class.

        Args:
            x: Single image tensor shaped (1, 3, 224, 224).
            class_idx: Target class index.

        Returns:
            Explanation dict with prototype index, map, score, and vector.

        Raises:
            ValueError: If batch size is not 1 or class_idx is out of range.
        """

        if x.shape[0] != 1:
            raise ValueError("get_explanation expects a batch size of 1.")
        if class_idx < 0 or class_idx >= self.num_classes:
            raise ValueError(f"class_idx out of range: {class_idx}")

        self.eval()
        logits, sim_maps, _ = self.forward(x, return_sim_maps=True)
        del logits

        class_proto_indices = torch.where(self.proto_class_map == class_idx)[0]
        class_maps = sim_maps[0, class_proto_indices]
        class_scores = class_maps.amax(dim=(1, 2))
        local_best = int(class_scores.argmax().item())
        proto_idx = int(class_proto_indices[local_best].item())
        spatial_map = sim_maps[0, proto_idx]
        sim_score = float(spatial_map.max().item())
        proto_vector = self.prototypes[proto_idx].detach().clone()

        return {
            "proto_idx": proto_idx,
            "spatial_map": spatial_map,
            "sim_score": sim_score,
            "proto_vector": proto_vector,
        }

    @torch.no_grad()
    def push_prototypes(self, dataloader: DataLoader, device: torch.device) -> None:
        """Push each prototype to its nearest training patch embedding.

        Args:
            dataloader: Training loader used for search.
            device: Active compute device.

        Returns:
            None.

        Raises:
            None.
        """

        self.eval()
        old_proto = self.prototypes.data.clone()
        best_dist = torch.full((self.total_proto,), float("inf"), device=device)
        best_embed = self.prototypes.data.clone().to(device)

        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            feats = self._extract_features(images)
            bsz, dim, height, width = feats.shape
            patches = feats.permute(0, 2, 3, 1).reshape(bsz * height * width, dim)
            dists = torch.cdist(patches, self.prototypes.to(device), p=2).pow(2)
            curr_min, curr_idx = dists.min(dim=0)

            improved = curr_min < best_dist
            if improved.any():
                best_dist = torch.where(improved, curr_min, best_dist)
                improved_indices = torch.where(improved)[0]
                best_embed[improved_indices] = patches[curr_idx[improved_indices]]

        self.prototypes.data.copy_(best_embed.to(self.prototypes.device))
        disp = (self.prototypes.data - old_proto).norm(dim=1).mean().item()
        print(f"Prototype push done. Mean displacement: {disp:.4f}")
        self.train()

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone and projection parameters.

        Args:
            freeze: Whether to freeze parameters.

        Returns:
            None.

        Raises:
            None.
        """

        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        for param in self.proj.parameters():
            param.requires_grad = not freeze

    def freeze_prototypes(self, freeze: bool = True) -> None:
        """Freeze or unfreeze prototype vectors.

        Args:
            freeze: Whether to freeze prototype parameters.

        Returns:
            None.

        Raises:
            None.
        """

        self.prototypes.requires_grad = not freeze