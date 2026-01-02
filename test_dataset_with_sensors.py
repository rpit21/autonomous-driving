from dataset.bddv_dataset import BDDVDataset, BDDVDatasetConfig

import json
import numpy as np
import cv2
from pathlib import Path
PLOT_DURATION_SEC = 40.0


ds = BDDVDataset(BDDVDatasetConfig(
    root_dir="data/bddv/t",
    clip_len=10,
))

TARGET_VIDEO = "0000f77c-cb820c98"

target_idx = None
for i, (video_path, start_frame) in enumerate(ds.index):
    if Path(video_path).stem == TARGET_VIDEO:
        target_idx = i
        break

if target_idx is None:
    raise RuntimeError(f"Video {TARGET_VIDEO} not found in dataset index")

print("Using dataset index:", target_idx)


frames, sensors, targets = ds[target_idx]


# --------------------------------------------------
# RECONSTRUCT STEERING (course_seq) FOR DEBUG
# --------------------------------------------------

# Get video path and start frame used by ds[0]
video_path, start_frame = ds.index[target_idx]


# Load corresponding JSON
video_stem = Path(video_path).stem
json_path = Path("data/bddv/t/info") / f"{video_stem}.json"

with open(json_path, "r") as f:
    info_json = json.load(f)

# Get FPS from video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# Compute timestamps for the clip frames
start_time_ms = info_json.get("startTime", 0)

# Number of frames covering ~40 seconds
n_frames_40s = int(PLOT_DURATION_SEC * fps)

frame_times_ms = []
for i in range(n_frames_40s):
    frame_idx = start_frame + i
    t_ms = start_time_ms + (frame_idx / fps) * 1000.0
    frame_times_ms.append(t_ms)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

max_frames_available = total_frames - start_frame
n_frames_40s = min(n_frames_40s, max_frames_available)

frame_times_ms = np.array(frame_times_ms)


# Extract GPS course
gps_data = info_json["locations"]
gps_ts = np.array([g["timestamp"] for g in gps_data], dtype=np.float64)
gps_course = np.deg2rad(
    np.array([g.get("course", 0.0) for g in gps_data], dtype=np.float32)
)
gps_speed = np.array(
    [g.get("speed", 0.0) for g in gps_data], dtype=np.float32
)


course_seq = []

last_valid = gps_course[0]
SPEED_MIN = 1.0  # m/s (ignore heading below this)

for t in frame_times_ms:
    j = np.argmin(np.abs(gps_ts - t))
    if gps_speed[j] > SPEED_MIN:
        last_valid = gps_course[j]
    course_seq.append(last_valid)

course_seq = np.unwrap(np.array(course_seq))



print(frames.shape)   # [K,3,224,224]
print(sensors.shape)  # [K,4]
print(targets.shape)  # [2]

acc_data = info_json["accelerometer"]
acc_ts = np.array([a["timestamp"] for a in acc_data])

ax = np.array([a["x"] for a in acc_data])
ay = np.array([a["y"] for a in acc_data])
az = np.array([a["z"] for a in acc_data])

ax_seq, ay_seq, az_seq = [], [], []

for t in frame_times_ms:
    j = np.argmin(np.abs(acc_ts - t))
    ax_seq.append(ax[j])
    ay_seq.append(ay[j])
    az_seq.append(az[j])

ax_seq = np.array(ax_seq)
ay_seq = np.array(ay_seq)
az_seq = np.array(az_seq)

time_sec = (frame_times_ms - frame_times_ms[0]) / 1000.0


# Visualize one clip

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(15,3))
for i in range(5):
    img = frames[i].permute(1,2,0).numpy()
    axes[i].imshow(img)
    axes[i].set_title(f"t={i}")
    axes[i].axis("off")
plt.show()

# --------------------------------------------------
# GYROSCOPE (RAW IMU)
# --------------------------------------------------

gyro_data = info_json.get("gyro", [])

if len(gyro_data) == 0:
    raise RuntimeError("No gyro data found in JSON")

gyro_ts = []
gyro_x = []
gyro_y = []
gyro_z = []

for g in gyro_data:
    # Skip malformed entries
    if "timestamp" not in g:
        continue
    gyro_ts.append(g["timestamp"])
    gyro_x.append(g.get("x", 0.0))
    gyro_y.append(g.get("y", 0.0))
    gyro_z.append(g.get("z", 0.0))

gyro_ts = np.array(gyro_ts, dtype=np.float64)
gyro_x = np.array(gyro_x, dtype=np.float32)
gyro_y = np.array(gyro_y, dtype=np.float32)
gyro_z = np.array(gyro_z, dtype=np.float32)

gyro_x_seq, gyro_y_seq, gyro_z_seq = [], [], []

for t in frame_times_ms:
    j = np.argmin(np.abs(gyro_ts - t))
    gyro_x_seq.append(gyro_x[j])
    gyro_y_seq.append(gyro_y[j])
    gyro_z_seq.append(gyro_z[j])

gyro_x_seq = np.array(gyro_x_seq)
gyro_y_seq = np.array(gyro_y_seq)
gyro_z_seq = np.array(gyro_z_seq)



# --------------------------------------------------
# STEERING VISUALIZATION
# --------------------------------------------------

# GPS course (heading)
plt.figure(figsize=(12,4))
plt.plot(time_sec, course_seq)
plt.xlabel("Time [s]")
plt.ylabel("Heading [rad]")
plt.title("GPS course over 40 seconds (speed-gated)")
plt.grid(True)
plt.show()

    

# Yaw rate (steering numerator)
yaw_rate = np.diff(course_seq, prepend=course_seq[0]) * fps
def moving_average(x, w=7):
    return np.convolve(x, np.ones(w) / w, mode="same")
yaw_rate_smooth = moving_average(yaw_rate, w=7)

plt.figure(figsize=(12,4))
plt.plot(time_sec, yaw_rate, alpha=0.4, label="raw yaw rate")
plt.plot(time_sec, yaw_rate_smooth, linewidth=2, label="smoothed yaw rate")
plt.xlabel("Time [s]")
plt.ylabel("Yaw rate [rad/s]")
plt.title("Steering signal over 40 seconds")
plt.legend()
plt.grid(True)
plt.show()



# Compare lateral accel with steering target
plt.figure(figsize=(12,4))
plt.plot(time_sec, az_seq, label="az (longitudinal accel)")
#plt.plot(time_sec, course_seq, linewidth=2, label="Heading [rad]")
plt.plot(time_sec, ay_seq, label="ay (lateral accel)")
plt.plot(time_sec, gyro_x_seq, label="Steering_gyro_x (roll)")
plt.legend()
plt.xlabel("Time [s]")
plt.title("Lateral acceleration vs steering (40s)_cb2fe290-88aacceb")
plt.grid(True)
plt.show()

# accelerometer raw
plt.figure(figsize=(12,4))
plt.plot(time_sec, ax_seq, label="ax")
plt.plot(time_sec, ay_seq, label="ay")
plt.plot(time_sec, az_seq, label="az")
plt.legend()
plt.xlabel("Time [s]")
plt.title("Accelerometer signals over 40 seconds")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(time_sec, gyro_x_seq, label="gyro_x (roll)")
plt.plot(time_sec, gyro_y_seq, label="gyro_y (pitch)")
plt.plot(time_sec, gyro_z_seq, label="gyro_z (yaw / steering)", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Angular velocity [rad/s]")
plt.title("Raw gyroscope signals (40s)")
plt.legend()
plt.grid(True)
plt.show()


