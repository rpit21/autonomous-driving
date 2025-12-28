from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Supported video extensions we will scan for inside root_dir
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class BDDVDatasetConfig:
    """
    Configuration object for the dataset.

    root_dir:
        Folder containing your videos.
    clip_len:
        Number of frames K per sample (temporal window size).
    stride:
        How much we move the starting frame between two consecutive samples.
        Example: stride=10 means sample starts at 0, 10, 20, ...
    frame_step:
        Spacing between frames inside a single clip.
        frame_step=1 -> consecutive frames
        frame_step=2 -> take every other frame, etc.
    resize_hw:
        Output image size (H, W). ViT often expects 224x224.
    to_rgb:
        OpenCV loads images as BGR by default; most deep models expect RGB.
    normalize_01:
        Convert uint8 [0..255] to float32 [0..1].
    return_uint8:
        If True, returns uint8 tensors (no normalization). Useful for debugging.
    min_video_frames:
        Ignore videos that are too short.
    """
    root_dir: str = "data/bddv/videos"
    clip_len: int = 10
    stride: int = 1
    frame_step: int = 1
    resize_hw: Tuple[int, int] = (224, 224)
    to_rgb: bool = True
    normalize_01: bool = True
    return_uint8: bool = False
    min_video_frames: int = 30


class BDDVDataset(Dataset):
    """
    BDDV Dataset (v1 - videos only, no labels yet).

    It returns clips of K frames from the videos. Each dataset item corresponds to:
        - a specific video file
        - a starting frame inside that video

    Return format:
        frames: Tensor of shape [K, 3, H, W]
        info: dict containing metadata useful for debugging
    """

    def __init__(self, cfg: BDDVDatasetConfig):
        self.cfg = cfg
        self.root = Path(cfg.root_dir)

        # 1) Validate that the root directory exists
        if not self.root.exists():
            raise FileNotFoundError(f"Folder not found: {self.root.resolve()}")

        # 2) Find all videos under root_dir
        self.videos: List[Path] = self._discover_videos(self.root)
        if len(self.videos) == 0:
            raise FileNotFoundError(f"No videos found under: {self.root.resolve()}")

        # 3) Pre-compute number of frames for each video (and filter short videos)
        self.video_num_frames: Dict[str, int] = {}
        for vp in self.videos:
            n = self._get_num_frames(vp)
            if n >= cfg.min_video_frames:
                self.video_num_frames[str(vp)] = n

        # Filter out short videos
        self.videos = [Path(v) for v in self.video_num_frames.keys()]
        if len(self.videos) == 0:
            raise RuntimeError(
                f"All videos have < {cfg.min_video_frames} frames. "
                f"Lower min_video_frames or check the video files."
            )

        # 4) Build a global index mapping each dataset item -> (video_path, start_frame)
        self.index: List[Tuple[str, int]] = self._build_index()

    def __len__(self) -> int:
        """
        Total number of clips in the dataset.
        """
        return len(self.index)

    def __getitem__(self, idx: int):
        """
        Return one sample:
            - frames: [K, 3, H, W]
            - info: metadata for debugging
        """
        video_path, start_frame = self.index[idx]
        frames = self._read_clip(video_path, start_frame)

        info = {
            "video_path": video_path,
            "start_frame": start_frame,
            "clip_len": self.cfg.clip_len,
            "frame_step": self.cfg.frame_step,
        }
        return frames, info

    # ------------------------ helper functions ------------------------

    def _discover_videos(self, root: Path) -> List[Path]:
        """
        Recursively scan root folder and collect all files with valid video extensions.
        """
        vids: List[Path] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                vids.append(p)
        vids.sort()
        return vids

    def _get_num_frames(self, video_path: Path) -> int:
        """
        Return total number of frames in a video using OpenCV metadata.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def _build_index(self) -> List[Tuple[str, int]]:
        """
        Build a list of (video_path, start_frame) pairs.
        Each pair corresponds to a possible clip of length K.

        Important detail:
            If frame_step > 1, the clip spans more than K frames in the original video.
            We compute 'needed' frames span as:
                needed = (K-1)*frame_step + 1
        """
        cfg = self.cfg
        index: List[Tuple[str, int]] = []

        # How many original frames are needed to extract K frames with step=frame_step
        needed = (cfg.clip_len - 1) * cfg.frame_step + 1

        for vp in self.videos:
            n = self.video_num_frames[str(vp)]

            # Last valid starting frame so that we can still read a full clip
            last_start = n - needed
            if last_start <= 0:
                continue

            # Generate multiple clips per video, sliding by 'stride'
            for start in range(0, last_start, cfg.stride):
                index.append((str(vp), start))

        if len(index) == 0:
            raise RuntimeError(
                "Index is empty. Check clip_len/frame_step/stride or video lengths."
            )
        return index
    
    def _read_clip(self, video_path: str, start_frame: int) -> torch.Tensor:
        """
        Read a clip of K frames starting from 'start_frame' in 'video_path'.

        Robust strategy:
        - Seek only once to start_frame
        - Then read sequentially (avoids OpenCV seek issues on some .mov files)
        """
        cfg = self.cfg
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Seek ONCE to the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frames_needed = cfg.clip_len

        # Read frames sequentially
        for _ in range(frames_needed):
            ok, frame_bgr = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(
                    f"Incomplete clip: requested {frames_needed} frames but got {len(frames)} "
                    f"from {video_path} at start_frame={start_frame}."
                )

            # If frame_step > 1, skip (frame_step - 1) frames using grab()
            # grab() is faster because it does not decode the frame fully.
            for _skip in range(cfg.frame_step - 1):
                ok_grab = cap.grab()
                if not ok_grab:
                    break

            # Convert BGR -> RGB if needed
            if cfg.to_rgb:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame = frame_bgr

            # Resize to (H, W)
            h, w = cfg.resize_hw
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            frames.append(frame)

        cap.release()

        # Stack to numpy: [K, H, W, 3]
        arr = np.stack(frames, axis=0)

        # Optionally return uint8 tensor (debug)
        if cfg.return_uint8:
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [K,3,H,W]
            return t

        # float32 tensor
        t = torch.from_numpy(arr).float()  # [K,H,W,3]
        if cfg.normalize_01:
            t = t / 255.0

        # [K,3,H,W]
        t = t.permute(0, 3, 1, 2).contiguous()
        return t

