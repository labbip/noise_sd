"""Unified, methodologically-consistent n-back serial dependence for the paper:
the SAME response-autocorrelation correction (Ma et al. 2025) is applied to the
MODEL and to the HUMAN data, via the identical code path in analyze_human_sd.

Consistency is achieved by turning each model into a human-format trial table:
for every 5-frame Gabor sequence the model's causal reconstruction of the last
frame is its "response" (err, bias), and its reconstruction of the 1-back frame
is its "response to the previous stimulus" (resp_rel_1bk) -- exactly the nuisance
regressor the human correction residualises. Model subject i -> one "subject".

Outputs (corrected == 'final' profile: 1-back uncorrected, lag>=2 corrected):
  results/table1_corrected.{json,md}          Table 1 (model vs human, corrected)
  figs/nback_model_a.pdf / _s.pdf             Fig 6 model (bars / DoG curves)
  figs/nback_human_a.pdf / _s.pdf             Fig 7a in-house human (bars / curves)
  figs/nback_human_pascucci_a.pdf / _s.pdf    Fig 7b public Pascucci (bars / curves)
  figs/nback_model_vs_human_corrected.pdf     Fig 8 model vs human bars
  results/model_trials_noise{sigma}.csv       the generated model trial table
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats

import config
from src.models.autoencoder import RecurrentAutoencoder
from src.utils.gabor import generate_gabor, estimate_orientation
from src.utils.dog import dog_func, fit_dog, get_angle_diff
from src.utils.save_results import save_json, save_markdown, fmt
import analyze_human_sd as A

NBACKS = [1, 2, 3]
FIG_SIZE = (6.4, 4.8)   # match the evaluate_* figure convention


# --------------------------- model -> human-format trials ---------------------------

def make_gabor_seq(orients):
    frames = [generate_gabor(theta=np.deg2rad(o), **config.GABOR) for o in orients]
    arr = np.stack(frames)[:, None, :, :].astype(np.float32)
    return torch.from_numpy(arr)


def _est(recon):
    return np.rad2deg(estimate_orientation(recon, **config.GABOR))


def model_trial_df(model, device, uid, n_seq=200, seq_len=5, noise_std=0.0, seed=0):
    """One model -> DataFrame with the same columns/sign conventions as the human
    CSVs (see convert_pascucci). Only est of the current and 1-back frames are
    needed (the correction uses resp_rel_1bk); rel_Nbk come from true orientations.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_seq):
        orients = rng.randint(0, 180, size=seq_len)
        seq = make_gabor_seq(orients).unsqueeze(0).to(device)
        if noise_std > 0:
            seq = torch.clamp(seq + torch.randn_like(seq) * noise_std, 0, 1)
        with torch.no_grad():
            x_hat, _ = model(seq)
        resp_cur = _est(x_hat[0, -1, 0].cpu().numpy())     # response to current
        resp_1bk = _est(x_hat[0, -2, 0].cpu().numpy())     # response to 1-back
        true_cur = float(orients[-1])
        row = {"uid": uid, "ori": true_cur, "resp": resp_cur,
               "err": get_angle_diff(true_cur, resp_cur),
               "resp_rel_1bk": get_angle_diff(true_cur, resp_1bk)}
        for n in NBACKS:
            row[f"rel_{n}bk"] = get_angle_diff(true_cur, float(orients[-1 - n]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_model_df(device, noise=0.3, n_seq=200):
    paths = sorted(glob.glob(os.path.join(config.SUBJECTS_DIR,
                                          f"subject_*_noise{noise:.1f}.pth")))
    dfs = []
    for i, path in enumerate(paths):
        model = RecurrentAutoencoder(use_memory=True).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        dfs.append(model_trial_df(model, device, uid=f"model_{i}",
                                  n_seq=n_seq, seed=i))
        print(f"  model {i+1}/{len(paths)}", end="\r", flush=True)
    print()
    return pd.concat(dfs, ignore_index=True), len(paths)


# --------------------------- figures ---------------------------

def _final_amps(prof):
    return prof["per_subject_final"]


def fig_bars_single(prof, out_path, color):
    """One dataset: final DoG amplitude bars per n-back (Fig 6 / Fig 7 style)."""
    plt = config.matplotlib_setup()
    fig, ax = plt.subplots(figsize=FIG_SIZE, clear=True)
    m, s = zip(*[A._mean_sem(prof["per_subject_final"][n]) for n in NBACKS])
    cmap = plt.get_cmap("tab10")
    ax.bar([str(n) for n in NBACKS], m, yerr=s, capsize=4,
           color=[cmap(j) for j in range(len(NBACKS))] if color is None else color)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("n-back"); ax.set_ylabel(r"SD amplitude ($^\circ$)")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    print(f"[fig] wrote {out_path}")


def fig_curves_single(prof, out_path):
    """One dataset: corrected DoG curves per n-back. The line is the group
    mean-parameter DoG (amplitude = reported per-subject mean, width = median
    per-subject b), so it matches the amplitude bars and does not chase noise in
    the aggregate curve."""
    plt = config.matplotlib_setup()
    fig, ax = plt.subplots(figsize=FIG_SIZE, clear=True)
    centers = prof["centers"]
    xx = np.linspace(-90, 90, 200)
    cmap = plt.get_cmap("tab10")
    for j, n in enumerate(NBACKS):
        mc = prof["curves_final"][n]
        c = cmap(j)
        if mc is not None:
            ax.plot(centers, mc, "o", color=c, ms=3, alpha=0.4)
        ab = prof["final_fit"][n]
        if ab is not None:
            ax.plot(xx, dog_func(xx, *ab), "-", color=c, label=f"{n}-back")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"Relative orientation ($^\circ$)")
    ax.set_ylabel(r"Bias ($^\circ$)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    print(f"[fig] wrote {out_path}")


def fig_bars_multi(profs, out_path):
    """Grouped final-amplitude bars for several datasets (label, prof, color)."""
    plt = config.matplotlib_setup()
    fig, ax = plt.subplots(figsize=FIG_SIZE, clear=True)
    x = np.arange(len(NBACKS))
    w = 0.8 / len(profs)
    for i, (label, prof, color) in enumerate(profs):
        m, s = zip(*[A._mean_sem(prof["per_subject_final"][n]) for n in NBACKS])
        ax.bar(x + (i - (len(profs) - 1) / 2) * w, m, w, yerr=s, capsize=4,
               label=label, color=color)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in NBACKS])
    ax.set_xlabel("n-back"); ax.set_ylabel(r"SD amplitude ($^\circ$)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight")
    print(f"[fig] wrote {out_path}")


# --------------------------- table ---------------------------

def table_rows(model_prof, human_prof, pascucci_prof):
    rows = []
    srcs = [("model", model_prof), ("human_inhouse", human_prof),
            ("human_pascucci", pascucci_prof)]
    for n in NBACKS:
        row = {"nback": n}
        for name, prof in srcs:
            v = prof["per_subject_final"][n]
            m, s = A._mean_sem(v)
            _, p = A._one_sided_p(v)
            row[f"{name}_mean"] = float(m)
            row[f"{name}_sem"] = float(s)
            row[f"{name}_p"] = float(p)
            row[f"{name}_n"] = int(np.isfinite(v).sum())
        rows.append(row)
    return rows


def save_table(rows, meta):
    save_json("table1_corrected.json", {"meta": meta, "table": rows},
              config.RESULTS_DIR)
    lines = [
        "# Table 1 (corrected) — n-back serial dependence, consistent method\n",
        "- Method: response-autocorrelation-corrected DoG amplitude "
        "(Ma et al. 2025), applied identically to model and human. "
        "'Final' profile = 1-back uncorrected + lag≥2 corrected (trial-level).",
        f"- Model: N={meta['model_n']} subject models (σ={meta['sigma']}), "
        f"{meta['n_seq']} sequences each; the model's per-frame reconstruction is "
        "its response.",
        f"- Human in-house: N={meta['human_n']}. "
        f"Human Pascucci 2019 (pooled Exp1+2): N={meta['pascucci_n']}.",
        "- p: one-sample t-test vs 0 (one-sided).\n",
        "| n-back | Model mean±SEM (°) | Model p | Human in-house mean±SEM (°) | "
        "p | Pascucci mean±SEM (°) | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['nback']} "
            f"| {fmt(r['model_mean'],3)}±{fmt(r['model_sem'],3)} "
            f"| {fmt(r['model_p'],3)} "
            f"| {fmt(r['human_inhouse_mean'],3)}±{fmt(r['human_inhouse_sem'],3)} "
            f"| {fmt(r['human_inhouse_p'],3)} "
            f"| {fmt(r['human_pascucci_mean'],3)}±{fmt(r['human_pascucci_sem'],3)} "
            f"| {fmt(r['human_pascucci_p'],3)} |")
    save_markdown("table1_corrected.md", "\n".join(lines) + "\n", config.RESULTS_DIR)
    print(f"[results] wrote {config.RESULTS_DIR}/table1_corrected.{{json,md}}")


# --------------------------- main ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--n_seq", type=int, default=200)
    p.add_argument("--human_csv", default=config.HUMAN_CSV)
    p.add_argument("--subj_excl", type=float, default=39.2)
    p.add_argument("--pascucci_csv",
                   default=os.path.join(config.DATA_DIR, "pascucci2019",
                                        "bhv_pascucci_pool12.csv"))
    p.add_argument("--reuse_model_csv", action="store_true",
                   help="reuse a previously saved model trial table")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.FIGS_DIR, exist_ok=True)
    model_csv = os.path.join(config.RESULTS_DIR,
                             f"model_trials_noise{args.noise:.1f}.csv")

    # ---- model trial table (identical human-format columns) ----
    if args.reuse_model_csv and os.path.exists(model_csv):
        print(f"Reusing {model_csv}")
        model_df = pd.read_csv(model_csv)
        n_models = model_df["uid"].nunique()
    else:
        print(f"Generating model trials (σ={args.noise}) ...", flush=True)
        model_df, n_models = build_model_df(device, noise=args.noise,
                                            n_seq=args.n_seq)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        model_df.to_csv(model_csv, index=False)
        print(f"  wrote {model_csv} ({len(model_df)} trials, {n_models} models)")

    # ---- corrected profiles via the identical code path ----
    model_prof = A.profiles_from_df(model_df, subj_excl=None)
    human_prof = A.dataset_profiles(args.human_csv, subj_excl=args.subj_excl)
    pasc_prof = A.dataset_profiles(args.pascucci_csv, subj_excl=None)

    # ---- figures (one file per dataset for the n-back panels) ----
    F = config.FIGS_DIR
    fig_bars_single(model_prof, os.path.join(F, "nback_model_a.pdf"), color=None)
    fig_curves_single(model_prof, os.path.join(F, "nback_model_s.pdf"))
    fig_bars_single(human_prof, os.path.join(F, "nback_human_a.pdf"), color=None)
    fig_curves_single(human_prof, os.path.join(F, "nback_human_s.pdf"))
    fig_bars_single(pasc_prof, os.path.join(F, "nback_human_pascucci_a.pdf"), color=None)
    fig_curves_single(pasc_prof, os.path.join(F, "nback_human_pascucci_s.pdf"))
    # combined comparison (bars): model vs the two human datasets
    fig_bars_multi(
        [(r"Model ($\sigma$=%g)" % args.noise, model_prof, "#55a868"),
         ("Human in-house", human_prof, "#c44e52"),
         ("Human Pascucci", pasc_prof, "#4c72b0")],
        os.path.join(F, "nback_model_vs_human_corrected.pdf"))

    # ---- table + console ----
    rows = table_rows(model_prof, human_prof, pasc_prof)
    print("\n" + "=" * 92)
    print("Table 1 (corrected, consistent method): n-back SD amplitude, "
          "mean±SEM, one-sided p")
    print("-" * 92)
    print(f"{'n':>3} | {'Model':>22} | {'Human in-house':>22} | {'Pascucci':>22}")
    for r in rows:
        print(f"{r['nback']:>3} | "
              f"{r['model_mean']:>8.3f}±{r['model_sem']:<6.3f} p={r['model_p']:<5.3f}| "
              f"{r['human_inhouse_mean']:>8.3f}±{r['human_inhouse_sem']:<6.3f} "
              f"p={r['human_inhouse_p']:<5.3f}| "
              f"{r['human_pascucci_mean']:>7.3f}±{r['human_pascucci_sem']:<6.3f} "
              f"p={r['human_pascucci_p']:<5.3f}")
    print("=" * 92)

    meta = {"method": "response-autocorrelation-corrected (Ma et al. 2025), "
                      "identical for model and human; final profile",
            "sigma": args.noise, "n_seq": args.n_seq, "model_n": int(n_models),
            "human_n": int(human_prof["n_subj"]),
            "pascucci_n": int(pasc_prof["n_subj"]),
            "model_trials_csv": model_csv}
    save_table(rows, meta)


if __name__ == "__main__":
    main()
