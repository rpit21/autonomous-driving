import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.driving_model import DrivingModel, DrivingModelConfig
from models.vit_encoder import ViTEncoderConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Dataset & DataLoader (1 small batch)
    ds_cfg = BDDVDatasetConfig(
        root_dir="data/bddv/videos",
        clip_len=10,
        stride=10,
        frame_step=1,
        resize_hw=(224, 224),
    )
    dataset = BDDVDataset(ds_cfg)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    frames, _ = next(iter(loader))
    frames = frames.to(device)

    # 2) Dummy targets (replace later with real steering/accel)
    # Shape: [B, 2]
    targets = torch.zeros(frames.size(0), 2).to(device)

    # 3) Model
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

    # 4) Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    # 5) Forward
    preds = model(frames)
    loss = criterion(preds, targets)

    print("Initial loss:", loss.item())

    # 6) Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 7) Forward again (same batch) to see if loss decreases
    preds2 = model(frames)
    loss2 = criterion(preds2, targets)

    print("Loss after one update:", loss2.item())


if __name__ == "__main__":
    main()
