"""Train N independent subject models at fixed sigma (Figures 6, 8 and Table 1).

Each model sees a different random subset of Ego videos (shuffle_videos=True with a
per-subject seed), simulating between-subject variability. The paper uses sigma=0.3
with N=100 models.

    models/subjects/subject_{i}_noise{sigma:.1f}.pth
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

import config
from src.data.charades_dataset import CharadesEgoDataset
from src.models.autoencoder import RecurrentAutoencoder


def train_subject(idx, noise_std, epochs, max_videos, seq_len, batch_size, lr,
                  device, num_workers=4):
    torch.manual_seed(1000 + idx)
    dataset = CharadesEgoDataset(
        csv_file=config.ANNOT_CSV,
        video_dir=config.VIDEO_DIR,
        seq_len=seq_len,
        noise_std=noise_std,
        ego_only=True,
        max_videos=max_videos,
        random_start=True,
        shuffle_videos=True,
        seed=idx,  # different subset per subject
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    model = RecurrentAutoencoder(use_memory=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            opt.zero_grad()
            x_hat, _ = model(noisy)
            loss = loss_fn(x_hat, clean)
            loss.backward()
            opt.step()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_models", type=int, default=100)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_videos", type=int, default=500)
    p.add_argument("--noise", type=float, default=0.3)   # paper: sigma=0.3
    p.add_argument("--seq_len", type=int, default=config.SEQ_LEN)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--start", type=int, default=0, help="resume from this index")
    args = p.parse_args()

    problems = [x for x in config.check_data() if "HUMAN" not in x]
    if problems:
        raise SystemExit("Data check failed:\n  " + "\n  ".join(problems))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.SUBJECTS_DIR, exist_ok=True)
    print(f"Device: {device}  n_models={args.n_models} noise={args.noise}")

    for i in range(args.start, args.n_models):
        out = os.path.join(config.SUBJECTS_DIR, f"subject_{i}_noise{args.noise:.1f}.pth")
        if os.path.exists(out):
            print(f"[skip] subject {i} exists", flush=True)
            continue
        t0 = time.time()
        model = train_subject(
            i, args.noise, args.epochs, args.max_videos, args.seq_len,
            args.batch_size, args.lr, device, args.num_workers,
        )
        torch.save(model.state_dict(), out)
        print(f"[saved] subject {i}/{args.n_models} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
