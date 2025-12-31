import cv2
import torch
import numpy as np
from tqdm import tqdm

from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig
from models.driving_model import DrivingModel, DrivingModelConfig
from models.vit_encoder import ViTEncoderConfig


# =========================
# Configuration
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_PATH = "data/bddv/val/videos/cb2fdf4d-4b15f621.mov"
OUTPUT_VIDEO = "overlay_output.mp4"

CLIP_LEN = 10
MAX_STEPS = 300  # we limited the video (≈10 seconds)


# =========================
# Dataset
# =========================
ds_cfg = BDDVDatasetConfig(
    root_dir="data/bddv/val",
    clip_len=CLIP_LEN,
    stride=1,
    frame_step=1,
    resize_hw=(224, 224),
)

dataset = BDDVDataset(ds_cfg)
num_steps = min(len(dataset), MAX_STEPS)


# =========================
# Model
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

model = DrivingModel(model_cfg).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=DEVICE))
model.eval()


# =========================
# OpenCV video IO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

print(f"Writing overlay video: {OUTPUT_VIDEO}")
print(f"Using {num_steps} steps (~{num_steps/30:.1f} seconds)")


# =========================
# Overlay loop
# =========================
with torch.no_grad():
    for idx in tqdm(range(num_steps)):

        # --- read video frame ---
        ret, frame = cap.read()
        if not ret:
            print("Video ended early.")
            break

        # --- get prediction from dataset ---
        try:
            frames, sensors, targets = dataset[idx]
        except Exception as e:
            print(f"Skipping idx {idx}: {e}")
            continue

        frames = frames.unsqueeze(0).to(DEVICE)
        sensors = sensors.unsqueeze(0).to(DEVICE)

        preds = model(frames, sensors)

        pred = preds[0].cpu().numpy()
        gt = targets.numpy()

        # --- overlay text ---
        y0 = 30
        dy = 25

        cv2.putText(
            frame,
            f"GT steer:   {gt[0]:+.3f}",
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Pred steer: {pred[0]:+.3f}",
            (20, y0 + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"GT accel:   {gt[1]:+.3f}",
            (20, y0 + 2 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Pred accel: {pred[1]:+.3f}",
            (20, y0 + 3 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        writer.write(frame)


cap.release()
writer.release()
print("Overlay video generation finished.")
