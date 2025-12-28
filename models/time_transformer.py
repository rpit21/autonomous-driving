from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TimeTransformerConfig:
    """
    Configuration for the temporal Transformer.

    input_dim:
        Dimension of each temporal token (e.g. 768 from ViT).
    hidden_dim:
        Internal embedding dimension used by the temporal Transformer.
    num_layers:
        Number of TransformerEncoder layers.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout probability inside the Transformer.
    use_positional_encoding:
        If True, add learnable temporal positional embeddings.
    """
    input_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    use_positional_encoding: bool = True
    max_seq_len: int = 64  # maximum supported K


class TimeTransformer(nn.Module):
    """
    Temporal Transformer that models dependencies across time.

    Input:
        x: Tensor of shape [B, K, D]
            - B: batch size
            - K: sequence length (temporal window)
            - D: input feature dimension (e.g. 768)

    Output:
        out: Tensor of shape [B, hidden_dim]
            Representation of the current time step.
    """

    def __init__(self, cfg: TimeTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Project input dimension (D) to temporal hidden dimension (H)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        # Optional learnable temporal positional embeddings
        if cfg.use_positional_encoding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, cfg.max_seq_len, cfg.hidden_dim)
            )
        else:
            self.pos_embed = None

        # Transformer encoder layer (PyTorch built-in)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,  # IMPORTANT: input is [B, K, H]
        )

        # Stack multiple encoder layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )

        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
          1) Project input features to hidden_dim
          2) Add temporal positional encoding (if enabled)
          3) Apply Transformer encoder
          4) Select the last temporal token as output
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input [B,K,D], got {x.shape}")

        B, K, D = x.shape
        if K > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length K={K} exceeds max_seq_len={self.cfg.max_seq_len}"
            )

        # 1) Project to temporal hidden dimension
        h = self.input_proj(x)  # [B, K, H]

        # 2) Add positional encoding (only first K positions)
        if self.pos_embed is not None:
            h = h + self.pos_embed[:, :K, :]

        # 3) Temporal self-attention
        h = self.encoder(h)  # [B, K, H]
        h = self.norm(h)

        # 4) Use the last token as representation of "now"
        out = h[:, -1, :]  # [B, H]

        return out
