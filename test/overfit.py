import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.driving_model import DrivingModel, DrivingModelConfig
from models.vit_encoder import ViTEncoderConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # 1) Dataset (VERY SMALL)
    # =========================
    ds_cfg = BDDVDatasetConfig(
        root_dir="data/bddv/videos",
        clip_len=10,
        stride=1000,   # VERY large stride -> very few clips
        frame_step=1,
        resize_hw=(224, 224),
    )

    dataset = BDDVDataset(ds_cfg)

    # Take ONLY ONE batch, always the same
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,   # IMPORTANT: no shuffle
        num_workers=0,
    )

    frames, _ = next(iter(loader))
    frames = frames.to(device)

    # =========================
    # 2) Fixed targets
    # =========================
    # Constant targets to memorize
    targets = torch.tensor(
        [[0.5, -0.2],
         [0.5, -0.2]],
        dtype=torch.float32,
        device=device,
    )

    # =========================
    # 3) Model
    # =========================
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

    # =========================
    # 4) Loss & Optimizer
    # =========================
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,   # higher LR to memorize fast
    )

    # =========================
    # 5) Overfitting loop
    # =========================
    num_steps = 300

    print("\nStarting overfitting...\n")

    for step in trange(num_steps):
        preds = model(frames)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"Step {step:03d} | "
                f"Loss: {loss.item():.6f} | "
                f"Preds: {preds.detach().cpu().numpy()}"
            )

    print("\nOverfitting finished.")


if __name__ == "__main__":
    main()
