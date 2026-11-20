"""Charades-Ego dataset: video -> noisy grayscale sequence, clean target."""

import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)  # avoid oversubscription under DataLoader workers


class CharadesEgoDataset(Dataset):
    def __init__(
        self,
        csv_file,
        video_dir,
        seq_len=16,
        noise_std=0.2,
        ego_only=True,
        max_videos=None,
        random_start=True,
        shuffle_videos=False,
        img_size=64,
        seed=None,
    ):
        self.video_dir = video_dir
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.random_start = random_start
        self.img_size = img_size

        df = pd.read_csv(csv_file)
        if ego_only and "egocentric" in df.columns:
            df = df[df["egocentric"] == "Yes"]
        ids = df["id"].astype(str).tolist()

        if shuffle_videos:
            rng = random.Random(seed)
            rng.shuffle(ids)

        # keep only IDs whose mp4 actually exists
        valid = []
        for vid in ids:
            path = os.path.join(video_dir, vid + ".mp4")
            if os.path.exists(path):
                valid.append(vid)
            if max_videos is not None and len(valid) >= max_videos:
                break
        self.video_ids = valid
        if len(self.video_ids) == 0:
            raise RuntimeError(f"No mp4 videos found in {video_dir} for the CSV IDs.")

    def __len__(self):
        return len(self.video_ids)

    def _load_sequence(self, path):
        cap = cv2.VideoCapture(path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames < self.seq_len:
            n_frames = 0  # fall back to re-reading

        if self.random_start and n_frames > self.seq_len:
            start = random.randint(0, n_frames - self.seq_len)
        else:
            start = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        while len(frames) < self.seq_len:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.img_size, self.img_size))
            frames.append(gray.astype(np.float32) / 255.0)
        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((self.img_size, self.img_size), np.float32)]
        while len(frames) < self.seq_len:  # pad by repeating last frame
            frames.append(frames[-1].copy())

        arr = np.stack(frames[: self.seq_len])[:, None, :, :]  # (T,1,H,W)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        path = os.path.join(self.video_dir, self.video_ids[idx] + ".mp4")
        clean = self._load_sequence(path)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean
