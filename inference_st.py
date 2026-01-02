import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.driving_model import DrivingModel, DrivingModelConfig
from models.vit_encoder import ViTEncoderConfig

def gyro_to_steering(gyro, r_scale=0.3):
    """
    Convert yaw rate (rad/s) to normalized steering command.
    """
    return np.tanh(gyro / r_scale)

# =========================
# Configuration
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "val_video_predictionst.npz"


# =========================
# Load dataset (NO shuffle)
# =========================
ds_cfg = BDDVDatasetConfig(
    root_dir="/homes/rmacias/data/t",
    clip_len=10,
    stride=1,          # temporal continuity
    frame_step=1,
    resize_hw=(224, 224),
)

dataset = BDDVDataset(ds_cfg)

print("Inference samples:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

# =========================
# Load model
# =========================
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

model = DrivingModel(model_cfg)
state_dict = torch.load("checkpoints/best.pt", map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)
model.to(DEVICE)
model.eval()


# =========================
# Storage
# =========================
pred_actions = []
gt_actions = []
timestamps = []

#On this configuration 1 step= 1 frame
MAX_STEPS = 1100   # or 200 it is like 6~7 seconds of video 
print(f"Using {MAX_STEPS} steps (~{MAX_STEPS/30:.1f} seconds)")

# =========================
# Inference loop
# =========================
with torch.no_grad():
    for idx, (frames, sensors, targets) in enumerate(
        tqdm(loader, total=MAX_STEPS)
    ):
        if idx >= MAX_STEPS:
            break

        frames = frames.to(DEVICE)
        sensors = sensors.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(frames, sensors)

        pred = preds[0].cpu().numpy()
        gt = targets[0].cpu().numpy()

        pred_actions.append(pred)
        gt_actions.append(gt)
        timestamps.append(idx)


# =========================
# Save results
# =========================
pred_actions = np.array(pred_actions)   # [T, 2]
gt_actions = np.array(gt_actions)
timestamps = np.array(timestamps)

np.savez(
    OUTPUT_FILE,
    pred_actions=pred_actions,
    gt_actions=gt_actions,
    timestamps=timestamps,
)

print(f"Saved predictions to {OUTPUT_FILE}")


# =========================
# Plot steering & acceleration
# =========================
plt.figure(figsize=(10, 6))

# --- Steering (gyro -> steering) ---
plt.subplot(2, 1, 1)

gt_steer   = gyro_to_steering(gt_actions[:, 0])    # GT gyro -> steering
pred_steer = gyro_to_steering(pred_actions[:, 0]) # Pred gyro -> steering

plt.plot(timestamps, gt_steer, label="GT gyro → steering")
plt.plot(timestamps, pred_steer, label="Pred gyro → steering")
plt.ylabel("Steering (normalized)")
plt.legend()
plt.grid(True)


# --- Acceleration ---
plt.subplot(2, 1, 2)
plt.plot(timestamps, gt_actions[:, 1], label="GT acceleration")
plt.plot(timestamps, pred_actions[:, 1], label="Pred acceleration")
plt.xlabel("Time step")
plt.ylabel("Acceleration")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("inference.png", dpi=150)
#plt.show()
plt.close()
