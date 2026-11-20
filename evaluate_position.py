"""Spatial position serial dependence (continuity field) -> Fig 8, 9."""

import argparse
import os

import numpy as np
import torch
from scipy import stats

import config
from src.models.autoencoder import RecurrentAutoencoder
from src.utils.blob import generate_blob, estimate_centroid
from src.utils.save_results import save_json, save_markdown, fmt

FIG_SIZE = (6.4, 4.8)


def bin_statistics(dists, projs, n_bins=14, dmax=28, alpha=0.01):
    """Per-bin mean/sem/n and 1-sample t-test vs 0 for a distance-bias curve."""
    edges = np.linspace(0, dmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(dists, edges) - 1
    out = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n > 2:
            vals = projs[sel]
            t, p = stats.ttest_1samp(vals, 0.0)
            out.append({"center_px": float(centers[b]), "n": n,
                        "bias_mean_px": float(vals.mean()),
                        "bias_sem_px": float(stats.sem(vals)),
                        "t": float(t), "p": float(p),
                        "significant": bool(p < alpha)})
        else:
            out.append({"center_px": float(centers[b]), "n": n,
                        "bias_mean_px": None, "bias_sem_px": None,
                        "t": None, "p": None, "significant": False})
    return out


def load_model(path, device):
    model = RecurrentAutoencoder(use_memory=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def position_bias(model, device, n_seq=5000, sigma_blob=8.0, noise_std=0.0,
                  lo=20, hi=44, seed=0, batch=200):
    """Return (distances, projected biases) over n_seq inducer/target pairs."""
    rng = np.random.RandomState(seed)
    dists, projs = [], []
    P1s = rng.uniform(lo, hi, size=(n_seq, 2))  # inducer (x,y)
    P2s = rng.uniform(lo, hi, size=(n_seq, 2))  # target (x,y)

    for start in range(0, n_seq, batch):
        idx = slice(start, min(start + batch, n_seq))
        p1b, p2b = P1s[idx], P2s[idx]
        seqs = []
        for p1, p2 in zip(p1b, p2b):
            f1 = generate_blob(64, (p1[0], p1[1]), sigma_blob)
            f2 = generate_blob(64, (p2[0], p2[1]), sigma_blob)
            seqs.append(np.stack([f1, f2])[:, None])
        seq = torch.from_numpy(np.stack(seqs).astype(np.float32)).to(device)
        if noise_std > 0:
            seq = torch.clamp(seq + torch.randn_like(seq) * noise_std, 0, 1)
        with torch.no_grad():
            x_hat, _ = model(seq)
        recon = x_hat[:, 1, 0].cpu().numpy()  # 2nd frame reconstruction
        for k, (p1, p2) in enumerate(zip(p1b, p2b)):
            cx, cy = estimate_centroid(recon[k])
            err = np.array([cx - p2[0], cy - p2[1]])
            d = p1 - p2
            dist = np.linalg.norm(d)
            if dist < 1e-6:
                continue
            u = d / dist
            proj = float(err @ u)  # + toward inducer, - repulsion
            dists.append(dist); projs.append(proj)
    return np.array(dists), np.array(projs)


def fig_position_comprehensive(all_results, out_path, n_bins=14, dmax=28, fig_num=0):
    plt = config.matplotlib_setup()
    fig, ax = plt.subplots(figsize=FIG_SIZE, num=fig_num, clear=True)
    edges = np.linspace(0, dmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # cmap = plt.get_cmap("viridis")
    cmap = plt.get_cmap("tab10")
    for jj, (sigma, (dists, projs)) in enumerate(all_results.items()):
        means = np.full(n_bins, np.nan)
        sig = np.zeros(n_bins, bool)
        idx = np.digitize(dists, edges) - 1
        for b in range(n_bins):
            sel = idx == b
            if sel.sum() > 2:
                vals = projs[sel]
                means[b] = vals.mean()
                _, pv = stats.ttest_1samp(vals, 0.0)
                sig[b] = pv < 0.01
        ax.plot(centers, means, "-", color=cmap(jj), label=rf"${sigma:.1f}$")
        ax.plot(centers[sig], means[sig], "o", color=cmap(jj))
        ax.plot(centers[~sig], means[~sig], "o", mfc="white", color=cmap(jj))
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"Distance $|P_1 - P_2|$ (px)")
    ax.set_ylabel("Bias toward inducer (px)")
    ax.legend(loc='upper right', title=r"Noise $\sigma$")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    # plt.close(fig)


def fig_flow(model, device, out_path, sigma_blob=8.0, noise_std=0.0, fig_num=0):
    """Qualitative: inducer / target / recon target / zoom for near & far configs."""
    plt = config.matplotlib_setup()
    configs = [("near", (32, 32), (39, 32)), ("far", (24, 32), (44, 32))]
    plt.close(fig_num)
    fig, axes = plt.subplots(len(configs), 3, figsize=(6, 4), num=fig_num, clear=True)
    for r, (name, p1, p2) in enumerate(configs):
        f1 = generate_blob(64, p1, sigma_blob)
        f2 = generate_blob(64, p2, sigma_blob)
        seq = torch.from_numpy(np.stack([f1, f2])[None, :, None].astype(np.float32)).to(device)
        if noise_std > 0:
            seq = torch.clamp(seq + torch.randn_like(seq) * noise_std, 0, 1)
        with torch.no_grad():
            x_hat, _ = model(seq)
        recon = x_hat[0, 1, 0].cpu().numpy()
        cx, cy = estimate_centroid(recon)
        axes[r, 0].imshow(f1, cmap="gray")
        # axes[r, 0].set_title(f"{name}: inducer")
        axes[r, 0].plot(p1[0], p1[1], "g.", ms=8, label="inducer")
        axes[r, 1].imshow(f2, cmap="gray")
        # axes[r, 1].set_title("target")
        axes[r, 1].plot(p2[0], p2[1], "b+", ms=10, label="true")
        axes[r, 2].imshow(recon, cmap="gray")
        axes[r, 2].plot(p1[0], p1[1], "g.", ms=8, label="inducer")
        axes[r, 2].plot(p2[0], p2[1], "b+", ms=10, label="true")
        axes[r, 2].plot(cx, cy, "rx", ms=10, label="est")
        # axes[r, 2].set_title("recon target")
        # axes[r, 3].imshow(recon, cmap="gray")
        # axes[r, 3].plot(cx, cy, "rx", ms=10, label="est")
        # axes[r, 2].set_title("overlay"); axes[r, 3].legend(fontsize=6)
        for c in range(3):
            axes[r, c].axis("off")
            axes[r, c].set_xlim([8, 64-8]); axes[r, c].set_ylim([64-8, 8])
    # fig.suptitle(rf"$\sigma={noise_std:.1f}$")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    # plt.close(fig)


def save_position_results(per_sigma_stats, args):
    """Persist the distance-bias (continuity field) curves and t-tests (Fig 8,9)."""
    data = {
        "meta": {
            "n_seq": args.n_seq, "blob_sigma_px": 8.0,
            "position_range_px": [20, 44], "distance_bins_px": [0, 28],
            "n_bins": 14, "significance_alpha": 0.01,
            "sign_convention": "+ = attraction toward inducer, - = repulsion",
            "figures": {
                "fig9_distance_bias": "figs/position_sd_comprehensive.pdf",
                "fig10_flow": "figs/vis_pos/flow_noise{sigma}.pdf",
            },
        },
        "per_sigma": {f"{s:.1f}": bins for s, bins in per_sigma_stats.items()},
    }
    save_json("position_results.json", data, config.RESULTS_DIR)

    lines = ["# Spatial position serial dependence — continuity field "
             "(Figures 8-9)\n",
             f"- {args.n_seq} inducer/target pairs per model; blob σ=8px; "
             "positions sampled from [20,44]px.",
             "- Bias projected onto the target→inducer direction: "
             "**+ = attraction, − = repulsion**.",
             "- Per distance bin: 1-sample t-test vs 0; significant if p<0.01.\n"]
    for s in sorted(per_sigma_stats):
        lines.append(f"\n## σ = {s:.1f}\n")
        lines.append("| dist (px) | n | bias mean (px) | sem | t | p | sig |")
        lines.append("|---|---|---|---|---|---|---|")
        for bn in per_sigma_stats[s]:
            sig = "*" if bn["significant"] else ""
            lines.append(
                f"| {bn['center_px']:.1f} | {bn['n']} | "
                f"{fmt(bn['bias_mean_px'],3)} | {fmt(bn['bias_sem_px'],3)} | "
                f"{fmt(bn['t'],2)} | {fmt(bn['p'],4)} | {sig} |")
    save_markdown("position_results.md", "\n".join(lines) + "\n", config.RESULTS_DIR)
    print(f"[results] wrote {config.RESULTS_DIR}/position_results.json + .md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_seq", type=int, default=5000)
    p.add_argument("--noise_grid", type=float, nargs="*", default=config.NOISE_GRID)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(config.FIGS_DIR, "vis_pos"), exist_ok=True)

    all_results = {}
    per_sigma_stats = {}
    for sigma in args.noise_grid:
        mpath = os.path.join(config.MODELS_DIR, f"model_ego_noise{sigma:.1f}.pth")
        if not os.path.exists(mpath):
            print(f"[warn] missing model {mpath}, skipping"); continue
        model = load_model(mpath, device)
        dists, projs = position_bias(model, device, n_seq=args.n_seq, noise_std=sigma)
        all_results[sigma] = (dists, projs)
        per_sigma_stats[sigma] = bin_statistics(dists, projs)
        fig_flow(model, device,
                 os.path.join(config.FIGS_DIR, "vis_pos", f"flow_noise{sigma:.1f}.pdf"),
                 noise_std=sigma, fig_num=4)
        # console summary at representative bins
        edges = np.linspace(0, 28, 15)
        idx = np.digitize(dists, edges) - 1
        near = projs[(idx >= 3) & (idx <= 4)]  # ~7px
        far = projs[(idx >= 12)]               # ~25px
        print(f"σ={sigma:.1f}  near(~7px) mean={near.mean():+.3f}px  "
              f"far(~25px) mean={far.mean():+.3f}px", flush=True)

    fig_position_comprehensive(
        all_results, os.path.join(config.FIGS_DIR, "position_sd_comprehensive.pdf"), fig_num=0)
    print("Saved figs/position_sd_comprehensive.pdf and flow visualizations.")

    save_position_results(per_sigma_stats, args)
