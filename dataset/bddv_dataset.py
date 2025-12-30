from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class BDDVDatasetConfig:
    root_dir: str = "data/bddv/train"   # or data/bddv/val
    videos_subdir: str = "videos"
    info_subdir: str = "info"
    clip_len: int = 10
    stride: int = 1
    frame_step: int = 1
    resize_hw: Tuple[int, int] = (224, 224)
    to_rgb: bool = True
    normalize_01: bool = True
    return_uint8: bool = False
    min_video_frames: int = 30


class BDDVDataset(Dataset):

    def __init__(self, cfg: BDDVDatasetConfig):

        self.cfg = cfg
        
        self.root = Path(cfg.root_dir)

        self.videos_dir = self.root / cfg.videos_subdir
        self.info_dir   = self.root / cfg.info_subdir

        if not self.videos_dir.exists():
            raise FileNotFoundError(self.videos_dir)

        if not self.info_dir.exists():
            raise FileNotFoundError(self.info_dir)

        # NOTE: placeholder normalization, to be refined if needed
        # Sensor normalization (ax, ay, az, |a|)
        self.sensor_mean = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.sensor_std  = torch.tensor([1.0, 1.0, 1.0, 0.5])

        # Info JSONs
        #Map video_id -> info.json
        self.video_to_info: Dict[str, Path] = {
            jp.stem: jp for jp in self.info_dir.glob("*.json")
        }

        # Discover Videos
        self.videos = self._discover_videos(self.videos_dir)

        self.video_num_frames = {}
        for vp in self.videos:
            n = self._get_num_frames(vp)
            if n >= cfg.min_video_frames:
                self.video_num_frames[str(vp)] = n

        self.videos = [Path(v) for v in self.video_num_frames]
        self.index = self._build_index()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        max_retries = 10
        last_err = None

        for _ in range(max_retries):
            video_path, start_frame = self.index[idx]

            try:
                frames = self._read_clip(video_path, start_frame)

                # Load JSON
                video_stem = Path(video_path).stem
                with open(self.video_to_info[video_stem], "r") as f:
                    info = json.load(f)

                # FPS
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps <= 0:
                    raise RuntimeError("Invalid FPS")

                # Frame timestamps
                start_time_ms = info.get("startTime", 0)
                frame_times_ms = [
                    start_time_ms + (start_frame + i * self.cfg.frame_step) / fps * 1000.0
                    for i in range(self.cfg.clip_len)
                ]

                # -------- ACCELEROMETER --------
                accel = info["accelerometer"]
                accel_ts = np.array([a["timestamp"] for a in accel])
                accel_xyz = np.array([[a["x"], a["y"], a["z"]] for a in accel])

                sensors_list = []
                ax_seq = []

                for t in frame_times_ms:
                    j = np.argmin(np.abs(accel_ts - t))
                    ax, ay, az = accel_xyz[j]
                    sensors_list.append([ax, ay, az, np.sqrt(ax*ax + ay*ay + az*az)])
                    ax_seq.append(ax)

                sensors = torch.tensor(sensors_list, dtype=torch.float32)
                sensors = (sensors - self.sensor_mean) / (self.sensor_std + 1e-6)

                ax_seq = np.array(ax_seq)

                # -------- GYRO (STEERING) --------
                gyro = info.get("gyro", [])
                if len(gyro) == 0:
                    raise RuntimeError("Missing gyro")

                gyro_ts = np.array([g["timestamp"] for g in gyro])
                gyro_x = np.array([g["x"] for g in gyro])  # yaw rate

                gyro_seq = []
                for t in frame_times_ms:
                    j = np.argmin(np.abs(gyro_ts - t))
                    gyro_seq.append(gyro_x[j])

                gyro_seq = np.array(gyro_seq)

                # Simple smoothing (no scipy)
                if len(gyro_seq) >= 7:
                    gyro_seq = np.convolve(
                        gyro_seq, np.ones(7)/7, mode="same"
                    )

                # -------- TARGETS --------
                steer_target = float(gyro_seq[-1])   # yaw-rate
                accel_target = float(ax_seq[-1])     # longitudinal accel

                targets = torch.tensor(
                    [steer_target, accel_target],
                    dtype=torch.float32
                )

                return frames, sensors, targets

            except Exception as e:
                last_err = e
                idx = torch.randint(0, len(self.index), (1,)).item()

        raise last_err

    # ---------------- helpers ----------------

    def _discover_videos(self, root: Path) -> List[Path]:
        return sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )

    def _get_num_frames(self, video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def _build_index(self):
        cfg = self.cfg
        index = []
        needed = (cfg.clip_len - 1) * cfg.frame_step + 1

        for vp in self.videos:
            n = self.video_num_frames[str(vp)]
            for start in range(0, n - needed, cfg.stride):
                index.append((str(vp), start))

        if not index:
            raise RuntimeError("Empty dataset index")

        return index

    def _read_clip(self, video_path: str, start_frame: int) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.cfg.clip_len):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Clip read failed")

            for _ in range(self.cfg.frame_step - 1):
                cap.grab()

            if self.cfg.to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(frame, self.cfg.resize_hw[::-1])
            frames.append(frame)

        cap.release()

        arr = np.stack(frames)
        t = torch.from_numpy(arr).float() / (255.0 if self.cfg.normalize_01 else 1.0)
        return t.permute(0, 3, 1, 2)
