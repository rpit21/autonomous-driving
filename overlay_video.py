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
MAX_STEPS = 300   # ≈ 10 seconds


# =========================
# Visualization parameters
# =========================
BAR_W = 220
BAR_H = 14

STEER_SCALE = 0.5    # [-0.5, 0.5]
ACC_SCALE = 2.0      # [-2, 2]


def error_color(err):
    if err < 0.05:
        return (0, 200, 0)      # green
    elif err < 0.15:
        return (0, 200, 200)    # yellow
    else:
        return (0, 0, 255)      # red


def draw_bar(frame, x, y, value, scale, color):
    center = x + BAR_W // 2
    length = int((value / scale) * (BAR_W // 2))
    length = max(-BAR_W // 2, min(BAR_W // 2, length))

    # outline
    cv2.rectangle(
        frame,
        (x, y),
        (x + BAR_W, y + BAR_H),
        (120, 120, 120),
        1,
    )

    # center line
    cv2.line(
        frame,
        (center, y),
        (center, y + BAR_H),
        (120, 120, 120),
        1,
    )

    # value bar
    cv2.rectangle(
        frame,
        (center, y),
        (center + length, y + BAR_H),
        color,
        -1,
    )


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

        ret, frame = cap.read()
        if not ret:
            print("Video ended early.")
            break

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

        # -------------------------
        # Compute errors
        # -------------------------
        steer_err = abs(pred[0] - gt[0])
        acc_err = abs(pred[1] - gt[1])

        steer_col = error_color(steer_err)
        acc_col = error_color(acc_err)

        # -------------------------
        # Text
        # -------------------------
        x_text = 20
        y_text = 30
        dy = 24

        cv2.putText(frame, f"GT steer:   {gt[0]:+.3f}",
                    (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.putText(frame, f"Pred steer: {pred[0]:+.3f}",
                    (x_text, y_text + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, steer_col, 2)

        cv2.putText(frame, f"GT accel:   {gt[1]:+.3f}",
                    (x_text, y_text + 2 * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.putText(frame, f"Pred accel: {pred[1]:+.3f}",
                    (x_text, y_text + 3 * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, acc_col, 2)

        # -------------------------
        # Bars
        # -------------------------
        x_bar = 20
        y_bar = y_text + 4 * dy + 10

        draw_bar(frame, x_bar, y_bar, gt[0], STEER_SCALE, (255, 255, 255))
        draw_bar(frame, x_bar, y_bar + BAR_H + 4, pred[0], STEER_SCALE, steer_col)

        cv2.putText(frame, "Steering (GT / Pred)",
                    (x_bar, y_bar - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_bar2 = y_bar + 2 * BAR_H + 35

        draw_bar(frame, x_bar, y_bar2, gt[1], ACC_SCALE, (255, 255, 255))
        draw_bar(frame, x_bar, y_bar2 + BAR_H + 4, pred[1], ACC_SCALE, acc_col)

        cv2.putText(frame, "Acceleration (GT / Pred)",
                    (x_bar, y_bar2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)


cap.release()
writer.release()
print("Overlay video generation finished.")
