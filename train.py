import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.driving_model import DrivingModel, DrivingModelConfig
from models.vit_encoder import ViTEncoderConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # Paths
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
        predict_dim=2,
        use_sensors=False,
        sensor_dim=0,
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
        lr=1e-4,
    )

    # ======================
    # Training loop
    # ======================
    num_epochs = 1 #3
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for frames, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            frames = frames.to(device)

            # Temporary dummy targets
            targets = torch.zeros(frames.size(0), 2, device=device)

            preds = model(frames)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"Epoch {epoch+1} - Avg loss: {epoch_loss:.6f}")

        # ======================
        # Save checkpoint
        # ======================
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (loss={best_loss:.6f})")

    print("Training finished.")


if __name__ == "__main__":
    main()
