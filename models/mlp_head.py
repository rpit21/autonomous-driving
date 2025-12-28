from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class MLPHeadConfig:
    """
    Configuration for the MLP regression head.

    input_dim:
        Dimension of the input feature vector (e.g. 256 from TimeTransformer).
    hidden_dim:
        Hidden dimension inside the MLP.
    output_dim:
        Number of regression outputs.
        Typically:
          - 1: steering only
          - 2: steering + acceleration
    dropout:
        Dropout probability.
    """
    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 2
    dropout: float = 0.1


class MLPHead(nn.Module):
    """
    Simple MLP head for regression.

    Input:
        x: Tensor of shape [B, input_dim]

    Output:
        y: Tensor of shape [B, output_dim]
    """

    def __init__(self, cfg: MLPHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP head.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected input [B,input_dim], got {x.shape}")

        return self.net(x)
