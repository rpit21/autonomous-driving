import os
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.driving_model import DrivingModel, DrivingModelConfig
from models.vit_encoder import ViTEncoderConfig


# ======================================================
# Utility: unpack batch safely
# ======================================================
def unpack_batch(
    batch: Any,
    device: torch.device,
    predict_dim: int,
    use_real_targets: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:

    if not isinstance(batch, (list, tuple)):
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    if len(batch) == 3:
        frames, sensors, targets = batch
    elif len(batch) == 4:
        frames, sensors, targets, _info = batch
    else:
        raise ValueError(f"Unsupported batch format with len={len(batch)}")

    frames = frames.to(device)
    sensors = sensors.to(device).float() if sensors is not None else None

    if targets is not None and use_real_targets:
        targets = targets.to(device).float()
        if targets.ndim != 2 or targets.shape[1] != predict_dim:
            raise ValueError(f"Targets must be [B,{predict_dim}], got {targets.shape}")
    else:
        B = frames.size(0)
        targets = torch.zeros(B, predict_dim, device=device)

    return frames, sensors, targets


# ======================================================
# Main training
# ======================================================
def main():

    # ======================
    # Configuration
    # ======================
    USE_REAL_TARGETS = True
    USE_SENSORS = True
    SENSOR_DIM = 4

    PREDICT_DIM = 2      # [steer, accel]
    NUM_EPOCHS = 5
    BATCH_SIZE = 2
    LR = 1e-4

    # Steering loss weight factor (MAIN FIX)
    STEER_ALPHA = 4.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # Checkpoints
    # ======================
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ======================
    # Dataset
    # ======================
    ds_cfg = BDDVDatasetConfig(
        root_dir="data/bddv/train",
        clip_len=10,
        stride=10,
        frame_step=1,
        resize_hw=(224, 224),
    )

    dataset = BDDVDataset(ds_cfg)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print(f"Training samples: {len(dataset)}")

    # ======================
    # Model
    # ======================
    model_cfg = DrivingModelConfig(
        predict_dim=PREDICT_DIM,
        use_sensors=USE_SENSORS,
        sensor_dim=SENSOR_DIM,
        vit=ViTEncoderConfig(
            model_name="google/vit-base-patch16-224-in21k",
            freeze=True,
            input_is_0_1=True,
        ),
    )

    model = DrivingModel(model_cfg).to(device)
    model.train()

    # ======================
    # Optimizer
    # ======================
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    # ======================
    # Training loop
    # ======================
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        ):

            frames, sensors, targets = unpack_batch(
                batch=batch,
                device=device,
                predict_dim=PREDICT_DIM,
                use_real_targets=USE_REAL_TARGETS,
            )

            preds = model(frames, sensors=sensors if USE_SENSORS else None)

            # ----------------------
            # Split predictions
            # ----------------------
            steer_pred = preds[:, 0]
            accel_pred = preds[:, 1]

            steer_gt = targets[:, 0]
            accel_gt = targets[:, 1]

            # ----------------------
            # Weighted steering loss
            # ----------------------
            steer_weight = 1.0 + STEER_ALPHA * torch.abs(steer_gt)

            # ---- logging steering weights (sanity check) ----
            if batch_idx % 200 == 0:
                print(
                    f"[Epoch {epoch+1} | Batch {batch_idx}] "
                    f"Avg steer weight: {steer_weight.mean().item():.2f} | "
                    f"Max steer weight: {steer_weight.max().item():.2f}"
                )

            steer_loss = steer_weight * (steer_pred - steer_gt) ** 2
            accel_loss = (accel_pred - accel_gt) ** 2

            loss = steer_loss.mean() + accel_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"\nEpoch {epoch+1} - Avg loss: {epoch_loss:.6f}")

        # ======================
        # Save checkpoints
        # ======================
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_ckpt)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (loss={best_loss:.6f})")

    print("Training finished.")


if __name__ == "__main__":
    main()
