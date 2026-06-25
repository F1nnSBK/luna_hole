"""
src/loss.py
===========
Loss functions for HOLE (Hole-Oriented LoRA Embedder).

Classes
-------
MatryoshkaTripletLoss
    Original Matryoshka Triplet Loss (cosine margin, unbounded gradient).
    Kept for backward compatibility and ablation experiments.

SelfLimitingHingeLoss
    DIVE-inspired hinge-based triplet loss that produces **exactly zero gradient**
    once a triplet already satisfies the margin constraint. Prevents unbounded
    gradient pressure on the pretrained embedding geometry.
    
    Reference: Zhao (2026) "DIVE: Embedding Compression via Self-Limiting
    Gradient Updates" arXiv:2605.20689

HeadwiseNTXentLoss
    Treats each of H projection heads as an implicit view of the same sample.
    Provides O(B·H²) dense pairwise contrastive gradients per batch, independent
    of triplet satisfaction — compensating for gradient sparsity in
    SelfLimitingHingeLoss on small datasets.

    Reference: Adapted from SimCLR / NT-Xent (Chen et al., 2020) and the
    multi-head contrastive objective in Zhao (2026).

MatryoshkaDIVELoss
    Synthesised loss combining SelfLimitingHingeLoss on the primary (full-dim)
    head with HeadwiseNTXentLoss across all H projection heads. Each head
    corresponds to a Matryoshka dimension, unifying DIVE + MRL in one objective.

    Loss = HingeTriplet(H₀, triplets) + λ_ntxent * NTXent(H₀...Hₙ)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Legacy loss (kept for ablations) ──────────────────────────────────────────

class MatryoshkaTripletLoss(nn.Module):
    """Cosine-margin Matryoshka Triplet Loss (original implementation).

    Computes triplet loss at each nested Matryoshka dimension and returns a
    weighted sum. Gradient is **unbounded** — active for every unsatisfied
    triplet throughout all epochs.

    Args:
        nest_dims: Sorted list of embedding sub-dimensions to evaluate.
        margin: Triplet margin (cosine distance space, typically 0.2–0.5).
        weights: Per-dimension loss weights (default: uniform 1.0).
    """

    def __init__(
        self,
        nest_dims: list[int] = [64, 128, 256, 384],
        margin: float = 0.3,
        weights: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.nest_dims = sorted(nest_dims)
        self.margin = margin
        self.weights = weights if weights is not None else [1.0] * len(nest_dims)

        distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.triplet_criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn,
            margin=self.margin,
        )

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = torch.tensor(0.0, device=anchor.device)
        losses_dict: dict[str, float] = {}

        for dim, weight in zip(self.nest_dims, self.weights):
            a_s = F.normalize(anchor[:, :dim], p=2, dim=1)
            p_s = F.normalize(positive[:, :dim], p=2, dim=1)
            n_s = F.normalize(negative[:, :dim], p=2, dim=1)

            step_loss = self.triplet_criterion(a_s, p_s, n_s)
            total_loss = total_loss + weight * step_loss
            losses_dict[f"loss_dim_{dim}"] = step_loss.item()

        return total_loss, losses_dict


# ── DIVE-inspired losses ───────────────────────────────────────────────────────

class SelfLimitingHingeLoss(nn.Module):
    """Hinge-based triplet loss with zero gradient for satisfied triplets.

    Unlike ``nn.TripletMarginWithDistanceLoss``, this loss produces *exactly*
    zero gradient once the margin condition is satisfied (d_an - d_ap ≥ margin).
    This bounds the total perturbation applied to the pretrained embedding space
    over the course of training.

    The loss per triplet is:
        ℓ = max(0, margin - (d_an - d_ap))

    where distances are in cosine space (d = 1 - cosine_similarity).

    A triplet with d_an - d_ap ≥ margin contributes zero loss AND zero gradient,
    so the fraction of active triplets (``rho``) naturally decays during training.

    Args:
        margin: Hinge margin in [0, 2] cosine distance space.
        reduction: "mean" | "sum" | "none".

    Returns (loss, metrics_dict) where metrics_dict contains:
        - "hinge_loss":    scalar loss value
        - "active_ratio":  fraction of triplets with non-zero gradient (ρ)
        - "mean_d_ap":     mean anchor–positive distance
        - "mean_d_an":     mean anchor–negative distance
    """

    def __init__(self, margin: float = 0.3, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        a = F.normalize(anchor, p=2, dim=1)
        p = F.normalize(positive, p=2, dim=1)
        n = F.normalize(negative, p=2, dim=1)

        d_ap = 1.0 - (a * p).sum(dim=1)  # cosine distance, shape (B,)
        d_an = 1.0 - (a * n).sum(dim=1)

        # Hinge: zero gradient for satisfied triplets (d_an - d_ap >= margin)
        raw = self.margin - (d_an - d_ap)
        per_triplet = torch.clamp(raw, min=0.0)

        if self.reduction == "mean":
            loss = per_triplet.mean()
        elif self.reduction == "sum":
            loss = per_triplet.sum()
        else:
            loss = per_triplet

        active = (per_triplet > 0).float()
        metrics = {
            "hinge_loss":   loss.item() if isinstance(loss, torch.Tensor) else float(loss),
            "active_ratio": active.mean().item(),
            "mean_d_ap":    d_ap.mean().item(),
            "mean_d_an":    d_an.mean().item(),
        }
        return loss, metrics


class HeadwiseNTXentLoss(nn.Module):
    """NT-Xent contrastive loss across H projection heads.

    Each pair of heads (Hᵢ, Hⱼ) for i ≠ j is treated as an implicit positive
    view pair for the same underlying sample. This yields O(B·H²) dense pairwise
    gradients per batch, independent of triplet satisfaction.

    For each batch of B samples with H heads:
    - Construct H×B normalized embeddings
    - For each head pair (i,j): compute SimCLR-style NT-Xent loss
    - Return the mean over all H(H-1)/2 pairs

    Args:
        temperature: NT-Xent softmax temperature (default: 0.07).
        
    Returns (loss, metrics_dict) with keys:
        - "ntxent_loss": scalar loss value
        - "n_pairs":     number of head pairs evaluated
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        heads: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            heads: List of H tensors, each shape (B, d_h). Each tensor is a
                   projected embedding from one projection head.
        """
        H = len(heads)
        if H < 2:
            zero = torch.tensor(0.0, device=heads[0].device, requires_grad=True)
            return zero, {"ntxent_loss": 0.0, "n_pairs": 0}

        # Normalise all heads
        normed = [F.normalize(h, p=2, dim=1) for h in heads]
        B = normed[0].shape[0]

        total_loss = torch.tensor(0.0, device=heads[0].device)
        n_pairs = 0

        for i in range(H):
            for j in range(i + 1, H):
                loss_ij = self._nt_xent_pair(normed[i], normed[j], B)
                total_loss = total_loss + loss_ij
                n_pairs += 1

        mean_loss = total_loss / n_pairs
        return mean_loss, {"ntxent_loss": mean_loss.item(), "n_pairs": n_pairs}

    def _nt_xent_pair(
        self,
        z_i: torch.Tensor,  # (B, d_i)
        z_j: torch.Tensor,  # (B, d_j) — may differ from d_i
        B: int,
    ) -> torch.Tensor:
        """NT-Xent loss between two views z_i and z_j (SimCLR formulation)."""
        d_min = min(z_i.shape[1], z_j.shape[1])
        z_i_proj = F.normalize(z_i[:, :d_min], p=2, dim=1)
        z_j_proj = F.normalize(z_j[:, :d_min], p=2, dim=1)

        # Concatenate: [z_i_proj; z_j_proj] → (2B, d_min)
        z = torch.cat([z_i_proj, z_j_proj], dim=0)

        # Pairwise similarity matrix (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarities
        mask = torch.eye(2 * B, device=z.device, dtype=z.dtype)
        sim = sim - mask * 1e9

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])  # (2B,)

        return F.cross_entropy(sim, labels)


@dataclass
class DIVELossConfig:
    """Configuration for MatryoshkaDIVELoss.
    
    Args:
        nest_dims: Matryoshka dimensions [d₀, d₁, d₂, d₃].
                   d₀ is the primary (full) dimension used for Hinge Triplet.
                   All dims are used for NT-Xent.
        hinge_margin: Margin for SelfLimitingHingeLoss (cosine distance space).
        ntxent_temperature: Temperature for HeadwiseNTXentLoss.
        lambda_ntxent: Weight of the NT-Xent term in the combined loss.
        hinge_weight: Weight of the Hinge Triplet term (default 1.0).
    """
    nest_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 384])
    hinge_margin: float = 0.3
    ntxent_temperature: float = 0.07
    lambda_ntxent: float = 0.5
    hinge_weight: float = 1.0


class MatryoshkaDIVELoss(nn.Module):
    """Unified DIVE + Matryoshka Representation Learning loss.

    Synthesises two complementary objectives:

    1. **SelfLimitingHingeLoss** on the primary (full-dimension) head:
       Bounds gradient pressure by producing zero gradient for satisfied triplets.
       Uses the full-dim embedding (nest_dims[-1]) as the primary retrieval space.

    2. **HeadwiseNTXentLoss** across all H Matryoshka heads:
       Treats each projection head as an implicit view of the same sample.
       Provides dense O(B·H²) contrastive gradients even with small batches,
       compensating for the sparsity of the hinge signal.

    The Matryoshka dimensions [d₀, d₁, ..., dₙ₋₁] are passed in as embedding
    slices — this module does NOT contain projection heads itself; instead it
    expects pre-projected head embeddings from a ``MatryoshkaProjectionHead``.

    Combined loss:
        L = w_hinge * HingeTriplet(heads[-1], a, p, n)
          + λ_ntxent * NTXentHeadwise([heads[0], ..., heads[-1]])

    Args:
        config: DIVELossConfig with all hyperparameters.

    Forward signature:
        heads:    List[Tensor(B, dₕ)] — one tensor per Matryoshka head,
                  ordered from smallest to largest dimension.
        anchor:   Tensor(B,)          — index into heads[-1] for anchors.
        positive: Tensor(B,)          — index into heads[-1] for positives.
        negative: Tensor(B,)          — index into heads[-1] for negatives.

    NOTE: anchor/positive/negative are the pre-selected triplet embeddings
    from heads[-1] (the full-dim primary head). Mining happens OUTSIDE this
    module (in the trainer) using the full-dim embeddings.

    Returns: (loss, metrics_dict)
    """

    def __init__(self, config: Optional[DIVELossConfig] = None) -> None:
        super().__init__()
        self.config = config or DIVELossConfig()
        self.hinge = SelfLimitingHingeLoss(margin=self.config.hinge_margin)
        self.ntxent = HeadwiseNTXentLoss(temperature=self.config.ntxent_temperature)

    @property
    def primary_dim(self) -> int:
        return max(self.config.nest_dims)

    def forward(
        self,
        heads: list[torch.Tensor],
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            heads:    H tensors (B, dₕ) — pre-projected Matryoshka embeddings,
                      ordered smallest→largest. heads[-1] is the primary head.
            anchor:   (N, d_primary) anchor embeddings for triplets.
            positive: (N, d_primary) positive embeddings.
            negative: (N, d_primary) negative embeddings.
        """
        # 1. Self-limiting hinge triplet on primary head
        hinge_loss, hinge_metrics = self.hinge(anchor, positive, negative)
        hinge_loss = self.config.hinge_weight * hinge_loss

        # 2. Head-wise NT-Xent across all Matryoshka heads
        ntxent_loss, ntxent_metrics = self.ntxent(heads)
        ntxent_loss = self.config.lambda_ntxent * ntxent_loss

        total = hinge_loss + ntxent_loss

        metrics = {
            "loss_total":    total.item(),
            "loss_hinge":    hinge_loss.item(),
            "loss_ntxent":   ntxent_loss.item(),
            **{f"hinge_{k}": v for k, v in hinge_metrics.items()},
            **{f"ntxent_{k}": v for k, v in ntxent_metrics.items()},
        }
        return total, metrics