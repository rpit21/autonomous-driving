import os
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.driving_model import DrivingModel, DrivingModelConfig
from models.vit_encoder import ViTEncoderConfig


def unpack_batch(
    batch: Any,
    device: torch.device,
    predict_dim: int,
    use_real_targets: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Unpack a batch from the DataLoader in a robust way.

    Supported batch formats:
      1) (frames, info)
      2) (frames, targets)
      3) (frames, sensors, targets)
      4) (frames, sensors, targets, info)  [optional]

    Returns:
      frames:  [B, K, 3, H, W]
      sensors: [B, K, S] or None
      targets: [B, predict_dim]
    """
    if not isinstance(batch, (list, tuple)):
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    if len(batch) == 2:
        frames, second = batch
        # Heuristic: if second is a Tensor -> targets, else -> info
        if torch.is_tensor(second):
            sensors = None
            targets = second
        else:
            sensors = None
            targets = None
    elif len(batch) == 3:
        frames, sensors, targets = batch
    elif len(batch) == 4:
        frames, sensors, targets, _info = batch
    else:
        raise ValueError(f"Unsupported batch format with len={len(batch)}")

    frames = frames.to(device)

    if sensors is not None:
        sensors = sensors.to(device).float()

    # Targets logic
    if targets is not None and use_real_targets:
        targets = targets.to(device).float()
        # Safety check
        if targets.ndim != 2 or targets.shape[1] != predict_dim:
            raise ValueError(f"Targets must be [B,{predict_dim}], got {targets.shape}")
    else:
        # Fallback dummy targets (debug mode)
        B = frames.size(0)
        targets = torch.zeros(B, predict_dim, device=device)

    return frames, sensors, targets


def main():
    # ======================
    # Main switches
    # ======================
    USE_REAL_TARGETS = False  # <-- change to True when your dataset returns real targets
    USE_SENSORS = False       # <-- change to True when your dataset returns sensors
    SENSOR_DIM = 0            # <-- set to S when USE_SENSORS=True (e.g., 2)

    PREDICT_DIM = 2           # steering + accel
    NUM_EPOCHS = 3
    LR = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # Checkpoints
    # ======================
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ======================
    # Dataset & DataLoader
    # ======================
    ds_cfg = BDDVDatasetConfig(
        root_dir="data/bddv/videos",
        clip_len=10,
        stride=10,
        frame_step=1,
        resize_hw=(224, 224),
    )

    dataset = BDDVDataset(ds_cfg)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

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
    # Loss & Optimizer
    # ======================
    criterion = nn.MSELoss()
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

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            frames, sensors, targets = unpack_batch(
                batch=batch,
                device=device,
                predict_dim=PREDICT_DIM,
                use_real_targets=USE_REAL_TARGETS,
            )

            # Forward (vision-only or vision+sensors)
            preds = model(frames, sensors=sensors if USE_SENSORS else None)

            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"Epoch {epoch+1} - Avg loss: {epoch_loss:.6f}")

        # Save epoch checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (loss={best_loss:.6f})")

    print("Training finished.")


if __name__ == "__main__":
    main()
