"""Train Ego-only models across the noise grid sigma in {0.0..0.5}.

Produces the six models used for Figures 2, 3, 4, 5, 9 and 10:
    models/model_ego_noise{sigma:.1f}.pth
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

import config
from src.data.charades_dataset import CharadesEgoDataset
from src.models.autoencoder import RecurrentAutoencoder


def train_one(noise_std, epochs, max_videos, seq_len, batch_size, lr, device,
              num_workers=4, seed=0):
    torch.manual_seed(1000 + seed)
    dataset = CharadesEgoDataset(
        csv_file=config.ANNOT_CSV,
        video_dir=config.VIDEO_DIR,
        seq_len=seq_len,
        noise_std=noise_std,
        ego_only=True,
        max_videos=max_videos,
        random_start=True,
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
        last = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            opt.zero_grad()
            x_hat, _ = model(noisy)
            loss = loss_fn(x_hat, clean)
            loss.backward()
            opt.step()
            last = loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    [sigma={noise_std:.1f}] epoch {epoch+1}/{epochs} "
                  f"loss={last:.5f}", flush=True)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--max_videos", type=int, default=config.MAX_VIDEOS)
    p.add_argument("--seq_len", type=int, default=config.SEQ_LEN)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--noise_grid", type=float, nargs="*", default=config.NOISE_GRID)
    args = p.parse_args()

    problems = [x for x in config.check_data() if "HUMAN" not in x]
    if problems:
        raise SystemExit("Data check failed:\n  " + "\n  ".join(problems))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    print(f"Device: {device}  noise_grid={args.noise_grid}")

    for sigma in args.noise_grid:
        out = os.path.join(config.MODELS_DIR, f"model_ego_noise{sigma:.1f}.pth")
        if os.path.exists(out):
            print(f"[skip] {out} exists", flush=True)
            continue
        t0 = time.time()
        print(f"=== Training sigma={sigma:.1f} ===", flush=True)
        model = train_one(
            sigma, args.epochs, args.max_videos, args.seq_len,
            args.batch_size, args.lr, device, args.num_workers,
            seed=int(round(sigma * 10)),
        )
        torch.save(model.state_dict(), out)
        print(f"[saved] {out} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
