from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class ViTEncoderConfig:
    """
    Configuration for the ViT encoder.

    model_name:
        Hugging Face model id for a pretrained ViT backbone.
        Common choices:
          - "google/vit-base-patch16-224-in21k"
          - "google/vit-base-patch16-224"
    freeze:
        If True, disables gradient updates for the ViT backbone (recommended at first).
    input_is_0_1:
        If True, the input frames are float in range [0, 1].
        If False, input is assumed to be uint8-like in [0, 255] (float or uint8).
    """
    model_name: str = "google/vit-base-patch16-224-in21k"
    freeze: bool = True
    input_is_0_1: bool = True


class ViTFrameEncoder(nn.Module):
    """
    Encodes a batch of video clips using a pretrained Vision Transformer (ViT).

    Input:
        frames: Tensor of shape [B, K, 3, H, W]
            - B = batch size
            - K = number of frames (temporal window length)
            - 3 channels (RGB)
            - H, W usually 224, 224 for ViT

    Output:
        cls_embeddings: Tensor of shape [B, K, D]
            - D is the hidden size of the ViT model (e.g., 768 for vit-base)
    """

    def __init__(self, cfg: ViTEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Load a pretrained ViT backbone (no classification head needed).
        # AutoModel returns the base model outputs including last_hidden_state.
        self.vit = AutoModel.from_pretrained(cfg.model_name)

        # Freeze backbone parameters if requested (recommended at the beginning).
        if cfg.freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        # ViT normalization constants (ImageNet standard).
        # We keep them as buffers so they move to GPU with the module.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Expose output dimension (hidden size D).
        self.hidden_size = self.vit.config.hidden_size

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Steps:
          1) Flatten [B, K, 3, H, W] -> [B*K, 3, H, W]
          2) Normalize using mean/std
          3) Run ViT
          4) Extract CLS token embedding (token index 0)
          5) Reshape back to [B, K, D]
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected frames with 5 dims [B,K,3,H,W], got shape {frames.shape}")

        B, K, C, H, W = frames.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels (RGB), got C={C}")

        # Flatten temporal dimension so ViT processes all frames as a single batch.
        x = frames.reshape(B * K, C, H, W)

        # Ensure float32 for the transformer.
        x = x.float()

        # If input is in [0,255], scale down to [0,1] first.
        if not self.cfg.input_is_0_1:
            x = x / 255.0

        # Normalize like ImageNet: (x - mean) / std
        x = (x - self.mean) / self.std

        # Run ViT backbone. Output includes last_hidden_state: [B*K, N+1, D]
        outputs = self.vit(pixel_values=x)
        tokens = outputs.last_hidden_state

        # CLS token is the first token (index 0): [B*K, D]
        cls = tokens[:, 0, :]

        # Reshape back to [B, K, D]
        cls = cls.reshape(B, K, self.hidden_size)

        return cls


def freeze_module(module: nn.Module) -> None:
    """Utility: freeze all parameters of a module."""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Utility: unfreeze all parameters of a module."""
    for p in module.parameters():
        p.requires_grad = True
