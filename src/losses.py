"""Loss definitions for ProtoCXR."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.model import LungMaskNet


class ARALoss(nn.Module):
    """Anatomical Region Alignment loss.

    Args:
        lung_net: Frozen lung mask network used to compute lung region masks.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self, lung_net: LungMaskNet) -> None:
        super().__init__()
        self.lung_net = lung_net

    def forward(self, sim_maps: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Compute L_ARA from similarity maps and lung masks.

        Args:
            sim_maps: Similarity maps shaped (B, C*K, 7, 7).
            images: Original images shaped (B, 3, 224, 224).

        Returns:
            Scalar ARA penalty tensor.

        Raises:
            ValueError: If sim_maps shape is invalid.
        """

        if sim_maps.dim() != 4:
            raise ValueError("sim_maps must be shaped (B, P, H, W).")

        bsz, num_proto, _, _ = sim_maps.shape
        with torch.no_grad():
            lung_mask = self.lung_net.get_7x7_mask(images)
        outside = 1.0 - lung_mask
        penalty = (sim_maps * outside).sum()
        return penalty / (bsz * num_proto)


class PDRLoss(nn.Module):
    """Prototype Diversity Regularizer.

    Args:
        sigma: Margin used by the hinge loss.
        num_classes: Number of classes C.
        num_proto: Number of prototypes per class K.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self, sigma: float, num_classes: int, num_proto: int) -> None:
        super().__init__()
        self.sigma = sigma
        self.num_classes = num_classes
        self.num_proto = num_proto

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute L_PDR without Python loops over classes/prototypes.

        Args:
            prototypes: Prototype tensor shaped (C*K, D).

        Returns:
            Scalar diversity regularizer tensor.

        Raises:
            ValueError: If prototype shape is inconsistent with C and K.
        """

        c_val = self.num_classes
        k_val = self.num_proto
        if k_val <= 1:
            return torch.zeros((), device=prototypes.device, dtype=prototypes.dtype)
        if prototypes.shape[0] != c_val * k_val:
            raise ValueError("prototypes shape does not match configured C*K.")

        proto = prototypes.view(c_val, k_val, -1)
        pairwise = torch.cdist(proto, proto, p=2)
        hinge = F.relu(self.sigma - pairwise)
        mask = 1.0 - torch.eye(k_val, device=prototypes.device).unsqueeze(0)
        hinge = hinge * mask
        denom = c_val * k_val * (k_val - 1)
        return hinge.sum() / denom


class SeparationLoss(nn.Module):
    """Prototype separation objective.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def forward(self, sim_maps: torch.Tensor) -> torch.Tensor:
        """Compute L_sep = -mean(sim_maps.amax(dim=(2,3))).

        Args:
            sim_maps: Similarity maps shaped (B, C*K, H, W).

        Returns:
            Scalar separation loss tensor.

        Raises:
            None.
        """

        return -sim_maps.amax(dim=(2, 3)).mean()


class ProtoCXRLoss(nn.Module):
    """Combined ProtoCXR loss wrapper.

    Args:
        lung_net: Frozen LungMaskNet instance.
        config: Global config carrying all loss hyperparameters.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self, lung_net: LungMaskNet, config: Config) -> None:
        super().__init__()
        self.lambda_ara = config.LAMBDA_ARA
        self.lambda_pdr = config.LAMBDA_PDR
        self.lambda_sep = config.LAMBDA_SEP

        self.bce = nn.BCEWithLogitsLoss()
        self.ara = ARALoss(lung_net)
        self.pdr = PDRLoss(
            sigma=config.PDR_SIGMA,
            num_classes=config.NUM_CLASSES,
            num_proto=config.NUM_PROTO,
        )
        self.sep = SeparationLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sim_maps: torch.Tensor,
        images: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss and all per-component values.

        Args:
            logits: Model logits shaped (B, C).
            labels: Binary targets shaped (B, C).
            sim_maps: Similarity maps shaped (B, C*K, 7, 7).
            images: Input images shaped (B, 3, 224, 224).
            prototypes: Prototype vectors shaped (C*K, D).

        Returns:
            Dict with keys total, bce, ara, pdr, sep.

        Raises:
            None.
        """

        bce_loss = self.bce(logits, labels)
        ara_loss = self.ara(sim_maps, images)
        pdr_loss = self.pdr(prototypes)
        sep_loss = self.sep(sim_maps)

        total = (
            bce_loss
            + self.lambda_ara * ara_loss
            + self.lambda_pdr * pdr_loss
            + self.lambda_sep * sep_loss
        )
        return {
            "total": total,
            "bce": bce_loss,
            "ara": ara_loss,
            "pdr": pdr_loss,
            "sep": sep_loss,
        }