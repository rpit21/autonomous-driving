from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.vit_encoder import ViTFrameEncoder, ViTEncoderConfig
from models.time_transformer import TimeTransformer, TimeTransformerConfig
from models.mlp_head import MLPHead, MLPHeadConfig


@dataclass
class DrivingModelConfig:
    """
    High-level configuration for the full driving model.

    predict_dim:
        Number of regression outputs.
        - 1: steering only
        - 2: steering + acceleration
    use_sensors:
        If True, the forward() expects sensors and concatenates them to ViT features.
    sensor_dim:
        Number of sensor features per frame (S). Example: speed,yaw_rate => S=2
    """
    predict_dim: int = 2
    use_sensors: bool = False
    sensor_dim: int = 0

    # Sub-modules configs
    vit: ViTEncoderConfig = ViTEncoderConfig()
    time: Optional[TimeTransformerConfig] = None  # filled in __post_init__
    head: Optional[MLPHeadConfig] = None          # filled in __post_init__

    # Temporal hidden dim (H)
    temporal_hidden_dim: int = 256

    def __post_init__(self):
        # Determine input dimension for the temporal transformer:
        # - vision only: D = 768
        # - vision + sensors: D = 768 + S
        vit_dim = 768  # default; will be overwritten when model is constructed
        input_dim = vit_dim + (self.sensor_dim if self.use_sensors else 0)

        # Create default configs for time transformer and head if not provided
        if self.time is None:
            self.time = TimeTransformerConfig(
                input_dim=input_dim,         # will be corrected again in model init
                hidden_dim=self.temporal_hidden_dim,
                num_layers=2,
                num_heads=4,
                dropout=0.1,
                use_positional_encoding=True,
                max_seq_len=64,
            )

        if self.head is None:
            self.head = MLPHeadConfig(
                input_dim=self.temporal_hidden_dim,
                hidden_dim=128,
                output_dim=self.predict_dim,
                dropout=0.1,
            )


class DrivingModel(nn.Module):
    """
    Full end-to-end driving model:

        frames [B,K,3,224,224]
          -> ViT CLS features [B,K,768]
          -> (optional concat sensors) [B,K,768+S]
          -> temporal transformer [B, H]
          -> MLP head [B, predict_dim]
    """

    def __init__(self, cfg: DrivingModelConfig):
        super().__init__()
        self.cfg = cfg

        # 1) Vision encoder
        self.vit = ViTFrameEncoder(cfg.vit)
        vit_dim = self.vit.hidden_size  # e.g. 768

        # 2) Temporal transformer input dimension depends on whether we add sensors
        temporal_input_dim = vit_dim + (cfg.sensor_dim if cfg.use_sensors else 0)

        # Ensure time transformer config matches the actual temporal input dim
        time_cfg = cfg.time
        time_cfg.input_dim = temporal_input_dim

        self.temporal = TimeTransformer(time_cfg)

        # 3) Regression head
        head_cfg = cfg.head
        # head input dim must match temporal hidden dim
        head_cfg.input_dim = time_cfg.hidden_dim
        head_cfg.output_dim = cfg.predict_dim

        self.head = MLPHead(head_cfg)

    def forward(self, frames: torch.Tensor, sensors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            frames: [B, K, 3, H, W]  (H=W=224 typically)
            sensors (optional): [B, K, S]

        Returns:
            y: [B, predict_dim]
        """
        # 1) Encode each frame with ViT (CLS token)
        vis = self.vit(frames)  # [B, K, 768]

        # 2) Optionally concatenate sensor features (per time step)
        if self.cfg.use_sensors:
            if sensors is None:
                raise ValueError("This model is configured with use_sensors=True, but sensors=None was provided.")
            if sensors.ndim != 3:
                raise ValueError(f"Expected sensors [B,K,S], got {sensors.shape}")

            # Concatenate on the last dimension: [B,K,768] + [B,K,S] => [B,K,768+S]
            x = torch.cat([vis, sensors.float()], dim=-1)
        else:
            # Vision-only
            x = vis

        # 3) Temporal transformer: [B,K,D] -> [B,H]
        ft = self.temporal(x)

        # 4) MLP head: [B,H] -> [B,predict_dim]
        y = self.head(ft)

        return y
