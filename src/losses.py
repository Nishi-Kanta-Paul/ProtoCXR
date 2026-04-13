"""
src/losses.py
=============
All loss functions for ProtoCXR as nn.Module classes:
  - ARALoss     : Anatomical Region Alignment
  - PDRLoss     : Prototype Diversity Regularizer
  - SeparationLoss : Encourages image patches to be close to a prototype
  - ProtoCXRLoss   : Combined weighted loss (BCE + ARA + PDR + Sep)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.model import LungMaskNet


class ARALoss(nn.Module):
    """Anatomical Region Alignment Loss.

    Penalises prototype similarity activations that fall *outside* the
    predicted lung mask, encouraging prototypes to focus on anatomically
    relevant regions.

    Formula::

        L_ARA = (1 / (B * P)) * Σ_{b,p,i,j} g(f_{b,i,j}, proto_p) * (1 - M_{b,i,j})

    where ``M`` is the binary lung mask at 7×7 resolution.

    Args:
        lung_net: Frozen :class:`LungMaskNet` instance.
    """

    def __init__(self, lung_net: LungMaskNet) -> None:
        super().__init__()
        self.lung_net = lung_net

    def forward(
        self,
        sim_maps: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ARA loss.

        Args:
            sim_maps: Similarity maps of shape ``(B, C*K, 7, 7)``.
            images:   Input images of shape ``(B, 3, 224, 224)`` used to
                      compute the lung mask.

        Returns:
            Scalar ARA loss tensor.
        """
        B, P, H, W = sim_maps.shape

        # Binary lung mask: (B, 1, 7, 7) — no gradients through LungMaskNet
        with torch.no_grad():
            mask = self.lung_net.get_7x7_mask(images)   # (B, 1, 7, 7)

        outside_mask = 1.0 - mask                        # (B, 1, 7, 7)
        # Broadcast to (B, P, 7, 7)
        penalty = (sim_maps * outside_mask).sum()
        # Normalise by batch size × num_prototypes to stabilise scale
        loss = penalty / (B * P)
        return loss


class PDRLoss(nn.Module):
    """Prototype Diversity Regularizer.

    Pushes same-class prototypes apart in embedding space using a hinge
    loss with a pre-defined margin ``sigma``.

    Formula::

        L_PDR = (1 / (C * K * (K-1))) *
                Σ_c Σ_{k≠k'} max(0, σ - ||p^c_k - p^c_{k'}||₂)

    Args:
        sigma:       Minimum desired L2 distance between same-class protos.
        num_classes: Number of classes ``C``.
        num_proto:   Number of prototypes per class ``K``.
    """

    def __init__(
        self,
        sigma: float,
        num_classes: int,
        num_proto: int,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.num_classes = num_classes
        self.num_proto = num_proto

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute PDR loss.

        Args:
            prototypes: Prototype parameter tensor of shape ``(C*K, D)``.

        Returns:
            Scalar PDR loss tensor. Returns zero if ``num_proto < 2``
            (no pairs to penalise).
        """
        if self.num_proto < 2:
            return torch.tensor(0.0, device=prototypes.device, requires_grad=False)

        C, K = self.num_classes, self.num_proto
        # Reshape → (C, K, D)
        p = prototypes.view(C, K, -1)

        # Pairwise L2 distances within each class: (C, K, K)
        # Using broadcasting: (C, K, 1, D) - (C, 1, K, D)
        diff = p.unsqueeze(2) - p.unsqueeze(1)          # (C, K, K, D)
        dist = diff.norm(dim=-1)                          # (C, K, K)

        # Hinge: max(0, σ - dist) — diagonal is 0 (self-distances)
        hinge = F.relu(self.sigma - dist)                # (C, K, K)

        # Zero out diagonal (k == k')
        eye = torch.eye(K, device=prototypes.device).unsqueeze(0)  # (1, K, K)
        hinge = hinge * (1.0 - eye)

        # Number of valid off-diagonal pairs per class
        n_pairs = C * K * (K - 1)
        loss = hinge.sum() / n_pairs
        return loss


class SeparationLoss(nn.Module):
    """Separation cost adapted from ProtoPNet.

    Encourages every image patch to be close to *at least one* prototype
    by maximising the peak similarity across all prototypes.

    Formula::

        L_sep = -mean_{b, p} max_{i,j} g(f_{b,i,j}, proto_p)
              = -mean(sim_maps.amax(dim=(2,3)))
    """

    def forward(self, sim_maps: torch.Tensor) -> torch.Tensor:
        """Compute separation loss.

        Args:
            sim_maps: ``(B, C*K, H, W)`` similarity activation maps.

        Returns:
            Scalar separation loss (negative mean of spatial max-pooled sims).
        """
        # Max-pool over spatial dims: (B, C*K)
        peak_sim = sim_maps.amax(dim=(2, 3))
        return -peak_sim.mean()


class ProtoCXRLoss(nn.Module):
    """Combined ProtoCXR loss.

    Computes::

        L = L_BCE + λ_ARA * L_ARA + λ_PDR * L_PDR + λ_sep * L_sep

    Each component is also returned individually for logging purposes.

    Args:
        lung_net:    Frozen :class:`LungMaskNet` for ARA computation.
        lambda_ara:  Weight of the ARA loss.
        lambda_pdr:  Weight of the PDR loss.
        lambda_sep:  Weight of the separation loss.
        sigma:       PDR margin.
        num_classes: Number of diagnostic classes.
        num_proto:   Number of prototypes per class.
    """

    def __init__(
        self,
        lung_net: LungMaskNet,
        lambda_ara: float,
        lambda_pdr: float,
        lambda_sep: float,
        sigma: float,
        num_classes: int,
        num_proto: int,
    ) -> None:
        super().__init__()
        self.lambda_ara = lambda_ara
        self.lambda_pdr = lambda_pdr
        self.lambda_sep = lambda_sep

        self.bce_fn  = nn.BCEWithLogitsLoss()
        self.ara_fn  = ARALoss(lung_net)
        self.pdr_fn  = PDRLoss(sigma=sigma,
                               num_classes=num_classes,
                               num_proto=num_proto)
        self.sep_fn  = SeparationLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sim_maps: torch.Tensor,
        images: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the total ProtoCXR loss and each component.

        Args:
            logits:     ``(B, C)`` raw model logits.
            labels:     ``(B, C)`` ground-truth binary label tensors.
            sim_maps:   ``(B, C*K, 7, 7)`` prototype similarity maps.
            images:     ``(B, 3, H, W)`` original input images.
            prototypes: ``(C*K, D)`` prototype parameter tensor.

        Returns:
            Dictionary with scalar tensors for keys:
            ``"total"``, ``"bce"``, ``"ara"``, ``"pdr"``, ``"sep"``.
        """
        bce = self.bce_fn(logits, labels)
        ara = self.ara_fn(sim_maps, images)
        pdr = self.pdr_fn(prototypes)
        sep = self.sep_fn(sim_maps)

        total = (
            bce
            + self.lambda_ara * ara
            + self.lambda_pdr * pdr
            + self.lambda_sep * sep
        )

        return {
            "total": total,
            "bce":   bce,
            "ara":   ara,
            "pdr":   pdr,
            "sep":   sep,
        }
