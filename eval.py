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
    # Dataset (NO SHUFFLE)
    # ======================
    ds_cfg = BDDVDatasetConfig(
        root_dir="data/bddv/val",
        clip_len=10,
        stride=10,
        frame_step=1,
        resize_hw=(224, 224),
    )

    dataset = BDDVDataset(ds_cfg)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,   # IMPORTANT for evaluation
        num_workers=0,
    )

    # ======================
    # Model
    # ======================
    model_cfg = DrivingModelConfig(
        predict_dim=2,
        use_sensors=True,
        sensor_dim=4,
        vit=ViTEncoderConfig(
            model_name="google/vit-base-patch16-224-in21k",
            freeze=True,
            input_is_0_1=True,
        ),
    )

    model = DrivingModel(model_cfg).to(device)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
    model.eval()

    # ======================
    # Loss (for reference)
    # ======================
    criterion = nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_samples = 0

    preds_all = []
    targets_all = []

    # ======================
    # Evaluation loop
    # ======================
    with torch.no_grad():
        for frames, sensors, targets in tqdm(loader):
            frames = frames.to(device)
            sensors = sensors.to(device)
            targets = targets.to(device)

            preds = model(frames, sensors)

            loss = criterion(preds, targets)
            total_loss += loss.item()
            total_samples += targets.size(0)

            preds_all.append(preds.cpu())
            targets_all.append(targets.cpu())

    avg_mse = total_loss / total_samples
    print(f"\nEvaluation MSE: {avg_mse:.6f}")

    # TODO (for evaluator):
    # - compute MAE
    # - compute RMSE
    # - plot predictions vs targets


if __name__ == "__main__":
    main()
