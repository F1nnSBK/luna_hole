"""
src/models.py
=============
Model components for HOLE (Hole-Oriented LoRA Embedder).

Classes
-------
DinoExtractor
    DINOv3 ViT backbone wrapped with a LoRA adapter (PEFT). Returns the raw
    [CLS] token embedding from the backbone — no projection head.

MatryoshkaProjectionHead
    Multi-head MLP projection on top of a backbone embedding. Implements the
    DIVE architecture: H independent 3-layer MLPs, one per Matryoshka dimension.
    Each head projects the backbone embedding to its target dimension.
    At inference, only the primary (full-dim) head is used.

HOLEModel
    Full HOLE model: DinoExtractor + MatryoshkaProjectionHead. Provides a clean
    training interface and handles adapter persistence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from .utils import get_logger

logger = get_logger(__name__)

# Default LoRA rank — same as before for backward compatibility
_DEFAULT_LORA_RANK = 32

# Embedding dim per model size
_DINO_EMBED_DIMS: dict[str, int] = {
    "vits16": 384,
    "vitb16": 768,
    "vitl16": 1024,
    "vitg16": 1536,
}


# ── Backbone ──────────────────────────────────────────────────────────────────

class DinoExtractor(nn.Module):
    """DINOv3 ViT backbone with a LoRA adapter.

    Loads a pretrained DINOv3 checkpoint, wraps it with PEFT/LoRA, and returns
    the [CLS] token embedding from the forward pass.

    Args:
        weights_path: Path to the pretrained ``.pth`` checkpoint.
        model_size: One of ``"vits16"``, ``"vitb16"``, ``"vitl16"``, ``"vitg16"``.
        lora_rank: LoRA rank ``r``. Also used as ``lora_alpha``.
        lora_dropout: Dropout in LoRA layers.
        lora_target_modules: Which attention sub-modules to apply LoRA to.
    """

    def __init__(
        self,
        weights_path: str = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth",
        model_size: str = "vits16",
        lora_rank: int = _DEFAULT_LORA_RANK,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] = ("qkv", "proj", "fc1", "fc2"),
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self._embed_dim = _DINO_EMBED_DIMS[model_size]

        logger.info(f"Loading DINOv3 backbone: {model_size} from {weights_path}")
        try:
            backbone = torch.hub.load(
                "facebookresearch/dinov3",
                f"dinov3_{model_size}",
                pretrained=False,
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            backbone.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            logger.error(f"Failed to load backbone: {exc}")
            raise

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=list(lora_target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.model = get_peft_model(backbone, lora_config)
        self.model.print_trainable_parameters()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [CLS] embedding of shape (B, embed_dim)."""
        return self.model(x)

    def save_adapter(self, path: str | Path) -> None:
        """Save LoRA adapter weights and patch the config for HF Hub compatibility."""
        path = Path(path)
        self.model.save_pretrained(str(path))

        config_path = path / "adapter_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config["base_model_name_or_path"] = "facebookresearch/dinov3"
            config["task_type"] = "FEATURE_EXTRACTION"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)


# ── Projection head ───────────────────────────────────────────────────────────

class _SingleProjectionHead(nn.Module):
    """3-layer MLP: in_dim → in_dim//2 → in_dim//4 → out_dim.
    
    Architecture follows DIVE (Zhao, 2026):
        Linear → BatchNorm1d → ReLU → Linear → BatchNorm1d → ReLU → Linear
    Xavier-uniform initialisation on all weight matrices.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        mid1 = max(out_dim * 2, in_dim // 2)
        mid2 = max(out_dim, in_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid1, bias=False),
            nn.BatchNorm1d(mid1),
            nn.ReLU(inplace=True),
            nn.Linear(mid1, mid2, bias=False),
            nn.BatchNorm1d(mid2),
            nn.ReLU(inplace=True),
            nn.Linear(mid2, out_dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MatryoshkaProjectionHead(nn.Module):
    """H independent projection heads, one per Matryoshka dimension.

    Each head is a 3-layer MLP (DIVE architecture) that maps the backbone
    embedding to its target dimension. Heads are independent — no shared
    parameters between them.

    The primary head (index ``primary_idx``) projects to the largest dimension
    and is used for triplet mining and hinge loss. All heads participate in the
    NT-Xent objective during training. At inference, only the primary head output
    is retained.

    Args:
        in_dim: Input dimension (backbone [CLS] embedding size).
        nest_dims: List of target output dimensions, ordered smallest→largest.
                   E.g. [64, 128, 256, 384].

    Forward returns a list of H tensors (B, dₕ), ordered smallest→largest.
    """

    def __init__(self, in_dim: int, nest_dims: list[int]) -> None:
        super().__init__()
        self.nest_dims = sorted(nest_dims)
        self.heads = nn.ModuleList([
            _SingleProjectionHead(in_dim, d) for d in self.nest_dims
        ])

    @property
    def primary_idx(self) -> int:
        """Index of the primary (largest-dim) head."""
        return len(self.nest_dims) - 1

    @property
    def primary_dim(self) -> int:
        return self.nest_dims[-1]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: Backbone embedding (B, in_dim).
        Returns:
            List of H tensors, each (B, dₕ), ordered smallest dim → largest.
        """
        return [head(x) for head in self.heads]

    def forward_primary(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the primary head output. Use at inference time."""
        return self.heads[self.primary_idx](x)


# ── Full model ────────────────────────────────────────────────────────────────

class HOLEModel(nn.Module):
    """Full HOLE model: DINOv3 LoRA backbone + Matryoshka projection heads.

    Combines DinoExtractor (frozen backbone + LoRA adapter) with a
    MatryoshkaProjectionHead to produce multi-scale embeddings.

    The forward pass returns all H head outputs for training. At inference,
    call ``embed()`` or ``forward_primary()`` to get only the primary embedding.

    Args:
        weights_path: Path to pretrained DINOv3 weights.
        model_size: DINOv3 variant (default "vits16").
        nest_dims: Matryoshka output dimensions.
        lora_rank: LoRA rank for the backbone adapter.
        lora_dropout: LoRA dropout.
    """

    def __init__(
        self,
        weights_path: str = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth",
        model_size: str = "vits16",
        nest_dims: list[int] = (64, 128, 256, 384),
        lora_rank: int = _DEFAULT_LORA_RANK,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = DinoExtractor(
            weights_path=weights_path,
            model_size=model_size,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
        )
        self.projection = MatryoshkaProjectionHead(
            in_dim=self.backbone.embed_dim,
            nest_dims=list(nest_dims),
        )
        self.nest_dims = list(sorted(nest_dims))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Training forward: returns list of H head embeddings (B, dₕ each).
        
        Embeddings are L2-normalised before return.
        """
        cls = self.backbone(x)                       # (B, embed_dim)
        heads = self.projection(cls)                  # [(B, d₀), ..., (B, dₙ)]
        return [F.normalize(h, p=2, dim=1) for h in heads]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: returns only the primary (full-dim) embedding, L2-normalised."""
        with torch.no_grad():
            cls = self.backbone(x)
            return F.normalize(self.projection.forward_primary(cls), p=2, dim=1)

    def save_adapter(self, path: str | Path) -> None:
        """Persist backbone LoRA adapter and projection head weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_adapter(path)
        torch.save(self.projection.state_dict(), path / "projection_head.pt")
        logger.info(f"Adapter + projection head saved to {path}")

    def load_projection(self, path: str | Path) -> None:
        """Load projection head weights from a saved checkpoint."""
        path = Path(path) / "projection_head.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.projection.load_state_dict(state)
        logger.info(f"Projection head loaded from {path}")