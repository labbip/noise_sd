"""Orientation serial dependence + reconstruction robustness -> Fig 2,3,4,5."""

import argparse
import os

import numpy as np
import torch
from scipy import stats

import config
from src.data.charades_dataset import CharadesEgoDataset
from src.models.autoencoder import RecurrentAutoencoder
from src.utils.gabor import generate_gabor, estimate_orientation
from src.utils.dog import (
    dog_func, fit_dog, get_angle_diff, bin_average, permutation_test_sd,
)
from src.utils.save_results import save_json, save_markdown, fmt
FIG_SIZE = (6.4, 4.8)


def load_model(path, device):
    model = RecurrentAutoencoder(use_memory=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def make_gabor_seq(orients):
    """orients: list of degrees -> tensor (T,1,64,64) in [0,1]."""
    frames = [generate_gabor(theta=np.deg2rad(o), **config.GABOR) for o in orients]
    arr = np.stack(frames)[:, None, :, :].astype(np.float32)
    return torch.from_numpy(arr)


def orientation_sd(model, device, n_seq=500, seq_len=5, noise_std=0.0, seed=0,
                   return_trials=False):
    """Return (rel_1back array, bias array, p, fit) over n_seq Gabor sequences."""
    rng = np.random.RandomState(seed)
    rels, biases = [], []
    for _ in range(n_seq):
        orients = rng.randint(0, 180, size=seq_len)
        seq = make_gabor_seq(orients).unsqueeze(0).to(device)  # (1,T,1,64,64)
        if noise_std > 0:
            seq = torch.clamp(seq + torch.randn_like(seq) * noise_std, 0, 1)
        with torch.no_grad():
            x_hat, _ = model(seq)
        recon = x_hat[0, -1, 0].cpu().numpy()
        est = np.rad2deg(estimate_orientation(recon, **config.GABOR))
        true_last = float(orients[-1])
        bias = get_angle_diff(est, true_last)
        rel = get_angle_diff(float(orients[-2]), true_last)
        rels.append(rel)
        biases.append(bias)
    rels = np.array(rels)
    biases = np.array(biases)
    p, fit = permutation_test_sd(rels, biases, n=1000)
    if return_trials:
        return rels, biases, p, fit
    return rels, biases, p, fit


def reconstruction_robustness(model, device, n_videos=100, seq_len=5, noise_std=0.2):
    """Per-video MSE for memory vs no-memory (reset hidden each frame)."""
    dataset = CharadesEgoDataset(
        csv_file=config.ANNOT_CSV, video_dir=config.VIDEO_DIR,
        seq_len=seq_len, noise_std=noise_std, ego_only=True,
        max_videos=n_videos, random_start=False, shuffle_videos=False,
    )
    mem_mse, nomem_mse = [], []
    for i in range(len(dataset)):
        noisy, clean = dataset[i]
        noisy = noisy.unsqueeze(0).to(device)   # (1,T,1,64,64)
        clean = clean.unsqueeze(0).to(device)
        with torch.no_grad():
            x_hat_mem, _ = model(noisy)  # full recurrent pass
            mem = ((x_hat_mem[:, -1] - clean[:, -1]) ** 2).mean().item()
            # no-memory: only the last frame, fresh hidden state
            last = noisy[:, -1:].contiguous()
            x_hat_no, _ = model(last, h=None)
            nomem = ((x_hat_no[:, -1] - clean[:, -1]) ** 2).mean().item()
        mem_mse.append(mem)
        nomem_mse.append(nomem)
    return np.array(mem_mse), np.array(nomem_mse)


# ------------------------- figures -------------------------

def fig_video_vis(model, device, out_path, seq_len=5, noise_std=0.2, fig_num=-1):
    plt = config.matplotlib_setup()
    dataset = CharadesEgoDataset(
        csv_file=config.ANNOT_CSV, video_dir=config.VIDEO_DIR,
        seq_len=seq_len, noise_std=noise_std, ego_only=True,
        max_videos=1, random_start=False,
    )
    noisy, clean = dataset[0]
    noisy_b = noisy.unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _ = model(noisy_b)
    recon = recon[0].cpu().numpy()
    # plt.close(fig_num)
    fig, axes = plt.subplots(3, seq_len, figsize=(2 * seq_len, 6), num=fig_num, clear=True)
    rows = [clean.numpy(), noisy.numpy(), recon]
    labels = ["ground-truth", "input", "recon"]
    for r in range(3):
        for c in range(seq_len):
            axes[r, c].imshow(rows[r][c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(labels[r])
                axes[r, c].axis("on")
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    # plt.close(fig)


def fig_gabor_vis(model, device, out_path, noise_std=0.0, fig_num=-1):
    plt = config.matplotlib_setup()
    orients = [0, 5, 15, 30, 60]
    # orients = np.random.randint(0, 180, size=5)
    # plt.close(fig_num)
    seq = make_gabor_seq(orients).unsqueeze(0).to(device)  # (1,T,1,64,64)
    if noise_std > 0:
        seq = torch.clamp(seq + torch.randn_like(seq) * noise_std, 0, 1)
    with torch.no_grad():
        x_hat, _ = model(seq)
    fig, axes = plt.subplots(2, len(orients), figsize=(2 * len(orients), 4.5), num=fig_num, clear=True)
    for c, o in enumerate(orients):
        recon = x_hat[0, c, 0].cpu().numpy()
        est = np.rad2deg(estimate_orientation(recon, **config.GABOR))
        axes[0, c].imshow(seq[0, c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        # axes[0, c].set_title(rf"{o}$^\circ$"); axes[0, c].axis("off")
        axes[0, c].text(2, 62, rf"{o}$^\circ$")
        axes[1, c].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[1, c].text(2, 62, rf"{est:.0f}$^\circ$")
        axes[0, c].axis("off")
        axes[1, c].axis("off")
        if c == 0:
            axes[0, c].set_ylabel("input")
            axes[0, c].axis("on")
            axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
            axes[1, c].set_ylabel("recon")
            axes[1, c].axis("on")
            axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    # plt.close(fig)


def fig_mse_mem(results, out_path, fig_num):
    plt = config.matplotlib_setup()
    sigmas = [r["sigma"] for r in results]
    mem = [r["mem_mean"] for r in results]
    nomem = [r["nomem_mean"] for r in results]
    mem_sem = [r["mem_sem"] for r in results]
    nomem_sem = [r["nomem_sem"] for r in results]
    x = np.arange(len(sigmas))
    w = 0.35
    fig, ax = plt.subplots(figsize=FIG_SIZE, num=fig_num, clear=True)
    ax.bar(x - w / 2, mem, w, yerr=mem_sem, label="Memory", capsize=3)
    ax.bar(x + w / 2, nomem, w, yerr=nomem_sem, label="No memory", capsize=3)
    for i, r in enumerate(results):
        if r["mse_p"] < 0.05:
            y = max(mem[i] + mem_sem[i], nomem[i] + nomem_sem[i]) * 1.01
            ax.text(i, y, "*", ha="center", fontsize=40)
    ax.set_xticks(x); ax.set_xticklabels([f"{s:.1f}" for s in sigmas])
    ax.set_xlabel(r"Training noise $\sigma$")
    ax.set_ylabel("Reconstruction MSE")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    # plt.close(fig)


def fig_sd_curves(results, out_path, fig_num):
    plt = config.matplotlib_setup()
    fig, ax = plt.subplots(figsize=FIG_SIZE, num=fig_num, clear=True)
    xx = np.linspace(-90, 90, 200)
    cmap = plt.get_cmap("tab10")
    for jj, r in enumerate(results):
        if r["fit"] is None:
            continue
        a, b = r["fit"]
        ax.plot(xx, dog_func(xx, a, b),
                # color=cmap(r["sigma"] / 0.5),
                color=cmap(jj),
                label=rf"{r['sigma']:.1f}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"Relative orientation of previous stimulus ($^\circ$)")
    ax.set_ylabel(r"Bias ($^\circ$)")
    ax.legend(title=rf"Noise $\sigma$", loc='lower right', ncols=1)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    # plt.close(fig)


def fig_sd_strength(results, out_path, fig_num):
    plt = config.matplotlib_setup()
    sigmas = [r["sigma"] for r in results]
    amps = [abs(r["fit"][0]) if r["fit"] else 0.0 for r in results]
    sig = [r["sd_p"] < 0.05 for r in results]
    fig, ax = plt.subplots(figsize=FIG_SIZE, num=fig_num, clear=True)
    ax.plot(sigmas, amps, "-o", color="k")
    for s, a, ok in zip(sigmas, amps, sig):
        ax.plot(s, a, "o", color="crimson" if ok else "gray", markersize=8)
    plt.xlim(-.01, .51)
    ax.set_xlabel(r"Training noise $\sigma$"); ax.set_ylabel(r"SD amplitude ($^\circ$)")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    # plt.close(fig)


def fig_sd_delta(results, out_path, fig_num):
    plt = config.matplotlib_setup()
    xs, ys = [], []
    for r in results:
        if r["fit"] and r["sd_p"] < 0.05:
            a, b = r["fit"]
            delta = 1.0 / (b * np.sqrt(2))
            xs.append(r["sigma"]); ys.append(delta)
    fig, ax = plt.subplots(figsize=FIG_SIZE, num=fig_num, clear=True)
    ax.plot(xs, ys, "-o", color="k")
    plt.xlim(-.01, .51)
    ax.set_xlabel(r"Training noise $\sigma$"); ax.set_ylabel(r"Peak position $\Delta$ ($^\circ$)")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    # plt.close(fig)


def save_orientation_results(results, args):
    """Persist orientation SD + reconstruction-robustness results (Fig 2-5)."""
    records = []
    for r in results:
        amp = abs(r["fit"][0]) if r["fit"] else None
        a = r["fit"][0] if r["fit"] else None
        b = r["fit"][1] if r["fit"] else None
        records.append({
            "sigma": r["sigma"],
            "sd_amplitude_deg": amp,               # |a|, SD strength (Fig 4b)
            "sd_amplitude_signed_deg": a,
            "sd_p_permutation": r["sd_p"],         # permutation test p (Fig 4)
            "dog_a": a, "dog_b": b,
            "delta_peak_deg": r["delta"],          # peak position (Fig 4c)
            "mse_mem_mean": r["mem_mean"], "mse_mem_sem": r["mem_sem"],
            "mse_nomem_mean": r["nomem_mean"], "mse_nomem_sem": r["nomem_sem"],
            "mse_paired_t": r["mse_t"], "mse_paired_p": r["mse_p"],  # Fig 3
        })
    meta = {"n_seq": args.n_seq, "n_videos_mse": args.n_videos,
            "figures": {
                "fig2_video_recon": "figs/vis/vis_video_model_ego_noise0.3.pdf",
                "fig3_mse_mem": "figs/mse_mem.pdf",
                "fig4a_sd_curves": "figs/sd_curve_combined_ego.pdf",
                "fig4b_sd_strength": "figs/sd_strength.pdf",
                "fig4c_sd_delta": "figs/sd_delta.pdf",
                "fig5_gabor_recon": "figs/vis/vis_gabor_model_ego_noise0.3.pdf",
            }}
    save_json("orientation_results.json",
              {"meta": meta, "per_sigma": records}, config.RESULTS_DIR)

    lines = ["# Orientation serial dependence & reconstruction robustness "
             "(Figures 2-5)\n",
             f"- Gabor sequences per model: {args.n_seq}; videos for MSE: "
             f"{args.n_videos}",
             "- `sd_amplitude_deg` = DoG amplitude |a| (SD strength, Fig 4b); "
             "significance via permutation test.",
             "- `delta_peak_deg` = DoG peak position Δ=1/(b√2) (Fig 4c).",
             "- MSE columns: memory vs no-memory reconstruction; paired t-test "
             "(Fig 3).\n",
             "| σ | SD amp (°) | SD p | Δ (°) | MSE Mem | MSE No-mem | paired t | "
             "MSE p |",
             "|---|---|---|---|---|---|---|---|"]
    for rec in records:
        lines.append(
            f"| {rec['sigma']:.1f} | {fmt(rec['sd_amplitude_deg'],3)} | "
            f"{fmt(rec['sd_p_permutation'],3)} | {fmt(rec['delta_peak_deg'],2)} | "
            f"{fmt(rec['mse_mem_mean'],5)} | {fmt(rec['mse_nomem_mean'],5)} | "
            f"{fmt(rec['mse_paired_t'],2)} | {rec['mse_paired_p']:.2e} |")
    save_markdown("orientation_results.md", "\n".join(lines) + "\n",
                  config.RESULTS_DIR)
    print(f"[results] wrote {config.RESULTS_DIR}/orientation_results.json + .md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_seq", type=int, default=500)
    p.add_argument("--n_videos", type=int, default=100)
    p.add_argument("--noise_grid", type=float, nargs="*", default=config.NOISE_GRID)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(config.FIGS_DIR, "vis"), exist_ok=True)

    results = []
    for sigma in args.noise_grid:
        mpath = os.path.join(config.MODELS_DIR, f"model_ego_noise{sigma:.1f}.pth")
        if not os.path.exists(mpath):
            print(f"[warn] missing model {mpath}, skipping"); continue
        model = load_model(mpath, device)
        rels, biases, sd_p, fit = orientation_sd(
            model, device, n_seq=args.n_seq, noise_std=0)
        mem, nomem = reconstruction_robustness(
            model, device, n_videos=args.n_videos, noise_std=sigma)
        tstat, mse_p = stats.ttest_rel(mem, nomem)
        delta = 1.0 / (fit[1] * np.sqrt(2)) if fit else np.nan
        results.append(dict(
            sigma=sigma, fit=fit, sd_p=sd_p, delta=delta,
            mem_mean=mem.mean(), nomem_mean=nomem.mean(),
            mem_sem=stats.sem(mem), nomem_sem=stats.sem(nomem),
            mse_t=tstat, mse_p=mse_p,
        ))
        # per-sigma visualizations for sigma=vis_sigma (Fig 2,5)
        vis_sigma = 0.3
        if abs(sigma - vis_sigma) < 1e-9:
            fig_video_vis(model, device,
                os.path.join(config.FIGS_DIR, "vis", f"vis_video_model_ego_noise{vis_sigma:.1f}.pdf"), noise_std=vis_sigma, fig_num=4)
            fig_gabor_vis(model, device,
                os.path.join(config.FIGS_DIR, "vis", f"vis_gabor_model_ego_noise{vis_sigma:.1f}.pdf"), noise_std=0, fig_num=5)

    # figures
    fig_mse_mem(results, os.path.join(config.FIGS_DIR, "mse_mem.pdf"), fig_num=0)
    fig_sd_curves(results, os.path.join(config.FIGS_DIR, "sd_curve_combined_ego.pdf"), fig_num=1)
    fig_sd_strength(results, os.path.join(config.FIGS_DIR, "sd_strength.pdf"), fig_num=2)
    fig_sd_delta(results, os.path.join(config.FIGS_DIR, "sd_delta.pdf"), fig_num=3)

    # console table
    print("\n" + "=" * 78)
    print(f"{'sigma':>6} {'SD_amp':>8} {'sd_p':>7} {'Delta':>7} "
          f"{'MSE_mem':>9} {'MSE_nomem':>10} {'t':>7} {'mse_p':>9}")
    print("-" * 78)
    for r in results:
        amp = abs(r["fit"][0]) if r["fit"] else float("nan")
        print(f"{r['sigma']:>6.1f} {amp:>8.3f} {r['sd_p']:>7.3f} "
              f"{r['delta']:>7.2f} {r['mem_mean']:>9.5f} {r['nomem_mean']:>10.5f} "
              f"{r['mse_t']:>7.2f} {r['mse_p']:>9.2e}")
    print("=" * 78)

    save_orientation_results(results, args)

