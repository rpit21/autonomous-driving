import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    state_dict = torch.load("checkpoints/best.pt", map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    # ======================
    # Loss (for reference)
    # ======================
    criterion = nn.MSELoss(reduction="mean")

    total_loss = 0.0
    num_batches = 0

    preds_all = []
    targets_all = []

    # ---- clip-wise accumulators ----
    clip_mae_all = []
    clip_rmse_all = []
    clip_max_all = []

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
            num_batches += 1

            preds_all.append(preds.cpu())
            targets_all.append(targets.cpu())

            # ======================
            # Clip-wise metrics
            # ======================
            error = preds - targets

            clip_mae = torch.mean(torch.abs(error), dim=1)
            clip_rmse = torch.sqrt(torch.mean(error ** 2, dim=1))
            clip_max = torch.max(torch.abs(error), dim=1).values

            clip_mae_all.append(clip_mae.cpu())
            clip_rmse_all.append(clip_rmse.cpu())
            clip_max_all.append(clip_max.cpu())

    avg_mse = total_loss / num_batches
    print(f"\nEvaluation MSE: {avg_mse:.6f}")

    # ======================
    # Aggregate clip-wise metrics
    # ======================
    clip_mae_all = torch.cat(clip_mae_all)
    clip_rmse_all = torch.cat(clip_rmse_all)
    clip_max_all = torch.cat(clip_max_all)

    print(f"Clip-wise MAE:  {clip_mae_all.mean().item():.6f}")
    print(f"Clip-wise RMSE: {clip_rmse_all.mean().item():.6f}")
    print(f"Clip-wise MAX:  {clip_max_all.mean().item():.6f}")

    # ======================
    # Aggregate metrics for dimension
    # ======================
    error_all = torch.cat(preds_all) - torch.cat(targets_all)

    mae_dim = torch.mean(torch.abs(error_all), dim=0)
    rmse_dim = torch.sqrt(torch.mean(error_all ** 2, dim=0))

    print(f"MAE steer:  {mae_dim[0]:.6f}")
    print(f"MAE accel:  {mae_dim[1]:.6f}")
    print(f"RMSE steer: {rmse_dim[0]:.6f}")
    print(f"RMSE accel: {rmse_dim[1]:.6f}")


    # ======================
    # Plot distribution of clip-wise RMSE
    # ======================
    plt.hist(clip_rmse_all.numpy(), bins=50)
    plt.title("Distribution of Clip-wise RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig("rmse_hist.png", dpi=150)
    #plt.show()
    plt.close()

    print("Saved rmse_hist.png")

    
    # ======================
    # Save evaluation outputs
    # ======================
    torch.save(
        {
            "preds": torch.cat(preds_all),
            "targets": torch.cat(targets_all),
        },
        "eval_outputs.pt",
    )


if __name__ == "__main__":
    main()
