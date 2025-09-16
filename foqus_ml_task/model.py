"""
MRI Embedding Model
===================

Purpose
-------
Compact, production-friendly CNN that maps multi-coil complex MRI inputs
(stack of real/imag channels from Task-1) to a fixed-size embedding vector.
This is designed for Task-2 requirements and to be directly usable in Task-3
contrastive learning (triplet/contrastive losses).

Input / Output
--------------
Input:
    x: Float32 tensor of shape (B, C_in, H, W), where C_in = 2 * num_coils
         (real and imaginary parts stacked channel-wise by Task-1 transforms)
Output:
    emb: Float32 tensor of shape (B, D) where D = embed_dim (L2-normalized
         by default for distance-based objectives)

Architecture Overview
---------------------
Stem:
    LazyConv2d -> GroupNorm -> GELU
Backbone (×4 stages):
    Conv2d(stride=2) -> GroupNorm -> GELU -> (optional Dropout)
Head:
    GlobalAveragePool -> Linear(proj to embed_dim) -> (optional 2-layer MLP)
    -> optional L2 normalization (on by default)

Design Choices
--------------
- **Simplicity & Speed:** Small CNN with stride-2 downsampling is stable and fast
  on synthetic phantoms; easy to read and maintain.
- **Batch-size Robustness:** GroupNorm instead of BatchNorm to behave well with
  small batches.
- **Flexible Input Channels:** LazyConv2d adapts to any coil count (C_in).
- **Contrastive-Ready:** L2-normalized embeddings make cosine/Euclidean distances
  well-behaved for triplet/contrastive losses in Task-3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

__all__ = ["MRIEmbeddingModel"]


def _groups_for(c: int, max_groups: int = 8) -> int:
    """
    Choose a GroupNorm group count that divides channels and is <= max_groups.
    Falls back to 1 (LayerNorm-like) if nothing else divides.
    """
    for g in range(min(max_groups, c), 0, -1):
        if c % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """
    A single CNN stage:
        Conv2d (stride=2) -> GroupNorm -> GELU -> (optional Dropout)

    Args:
        in_ch:   input channels
        out_ch:  output channels
        stride:  spatial stride (default: 2 for downsampling)
        p_drop:  dropout prob after activation (0 disables)
    """

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 2, p_drop: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(_groups_for(out_ch), out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop) if p_drop and p_drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MRIEmbeddingModel(nn.Module):
    """
    Compact CNN that maps multi-coil complex MRI images (stacked real/imag) to a
    fixed-size embedding, suitable for contrastive learning.

    Expected input:
        x: Float tensor of shape (B, C_in, H, W), where C_in = 2 * num_coils
    Output:
        (B, embed_dim), L2-normalized if `normalize=True`.

    Args:
        embed_dim:   output embedding size D (default: 256)
        widths:      channel widths per stage (default: (32, 64, 128, 256))
        dropout:     dropout prob inside ConvBlocks and MLP head (default: 0.0)
        normalize:   apply L2 norm to embeddings in forward() (default: True)
        use_mlp_head:add a 2-layer MLP projection head (default: False)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
        normalize: bool = True,
        use_mlp_head: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.use_mlp_head = use_mlp_head

        # Stem: adapt to arbitrary input channels (2 * num_coils)
        # LazyConv2d infers in_channels on first forward
        self.stem = nn.Sequential(
            nn.LazyConv2d(out_channels=widths[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups_for(widths[0]), widths[0]),
            nn.GELU(),
        )

        # Stages
        blocks = []
        for i in range(len(widths) - 1):
            blocks.append(ConvBlock(widths[i], widths[i + 1], stride=2, p_drop=dropout))
        self.backbone = nn.Sequential(*blocks)

        # Head: GAP -> Linear (optionally MLP)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        last_w = widths[-1]
        self.proj = nn.Linear(last_w, embed_dim)

        if self.use_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout if dropout else 0.0),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        else:
            self.head = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for convs (when materialized), Xavier for linears
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.LazyConv2d):
                w = getattr(m, "weight", None)
                # Skip if LazyConv2d weight is still uninitialized (materializes on first forward)
                if w is not None and not isinstance(w, UninitializedParameter):
                    nn.init.kaiming_normal_(w, nonlinearity="relu")  # ReLU gain works well for GELU
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, *, normalize: bool | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:          (B, C_in, H, W) tensor
            normalize:  override constructor's L2-normalization flag

        Returns:
            (B, embed_dim) tensor, L2-normalized if enabled
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape {tuple(x.shape)}")

        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, C_last)
        x = self.proj(x)
        x = self.head(x)

        do_norm = self.normalize if normalize is None else normalize
        if do_norm:
            x = F.normalize(x, p=2, dim=1)
        return x

    @torch.no_grad()
    def embed(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Convenience method to produce embeddings (no grad)."""
        return self.forward(x, normalize=normalize)
