"""n-back serial dependence on a human-format CSV, with and without the
response-autocorrelation correction of Ma, Yan et al. (2025),
"Investigating the effect of response autocorrelation on n-back analyses of
serial dependence".

Motivation: response autocorrelation can inflate 2-back+ n-back estimates and
manufacture a spurious non-monotonic (e.g. 2-back-peak) profile.  The
correction residualises out the 1-back *response* effect before testing the
n-back *stimulus* effect:

    y   = -err                         (response - truth, + = attraction)
    x_r = -resp_rel_1bk                (previous response - current)
    res = y - DoG(x_r ; fit)           (remove 1-back response serial dependence)
    A_n = amplitude of DoG(-rel_Nbk -> res)   (corrected n-back stimulus SD)

Runs on the in-house bhv_hum.csv and on the converted Pascucci et al. (2019)
public data (see convert_pascucci.py), so the two are directly comparable.

Usage:
    uv run python analyze_human_sd.py --csv data/bhv_hum.csv --subj_excl 39.2 --tag human
    uv run python analyze_human_sd.py --csv data/pascucci2019/bhv_pascucci_exp1.csv --tag pascucci_exp1
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.utils.dog import dog_func, fit_dog, bin_average
from src.utils.save_results import save_json, save_markdown, fmt

NBACKS = [1, 2, 3]
TRIAL_EXCL = 37.5
MIN_TRIALS = 30


def _is_monotone(means):
    """True if the sequence is (weakly) monotonically decreasing."""
    m = [x for x in means if np.isfinite(x)]
    return all(m[i] >= m[i + 1] for i in range(len(m) - 1))


def _one_sided_p(vals):
    """One-sample t-test vs 0, one-sided in the direction of the mean."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan, np.nan
    t, p2 = stats.ttest_1samp(vals, 0.0)
    p = p2 / 2 if np.mean(vals) > 0 else 1 - p2 / 2
    return float(t), float(p)


def subject_fits(g):
    """Return dict n -> (fit_uncorrected, fit_corrected) for one subject, where
    each fit is the full (a, b) DoG parameter tuple from fit_dog (or None).
    fit_corrected is None for n==1 (not response-correctable). Returns None if
    there are too few usable trials."""
    r = g[["resp_rel_1bk", "err"]].dropna()
    if len(r) < MIN_TRIALS:
        return None
    resp_fit = fit_dog(-r["resp_rel_1bk"].values.astype(float),
                       -r["err"].values.astype(float))
    out = {}
    for n in NBACKS:
        col = f"rel_{n}bk"
        sub = g[[col, "err", "resp_rel_1bk"]].dropna()
        if len(sub) < MIN_TRIALS:
            out[n] = (None, None)
            continue
        x = -sub[col].values.astype(float)
        y = -sub["err"].values.astype(float)
        f_unc = fit_dog(x, y)
        # Autocorrelation correction (Ma et al. 2025) applies only to lag >= 2:
        # the 1-back stimulus is collinear with the 1-back response, so
        # residualising the response would remove the genuine 1-back effect.
        if n == 1 or resp_fit is None:
            f_cor = None
        else:
            xr = -sub["resp_rel_1bk"].values.astype(float)
            f_cor = fit_dog(x, y - dog_func(xr, *resp_fit))
        out[n] = (f_unc, f_cor)
    return out


def analyze_subject(g):
    """Return dict n -> (amp_uncorrected, amp_corrected) for one subject, or
    None if there are too few usable trials."""
    fits = subject_fits(g)
    if fits is None:
        return None
    out = {}
    for n in NBACKS:
        f_unc, f_cor = fits[n]
        out[n] = (f_unc[0] if f_unc else np.nan,
                  f_cor[0] if f_cor else np.nan)
    return out


def binned_curve(g, n):
    col = f"rel_{n}bk"
    sub = g[[col, "err"]].dropna()
    if len(sub) < MIN_TRIALS:
        return None
    x = -sub[col].values.astype(float)
    y = -sub["err"].values.astype(float)
    _, means = bin_average(x, y, 18)
    return means


def subject_curves(g):
    """Return {n: (curve_uncorrected, curve_final)} binned bias curves for one
    subject. curve_final residualises the 1-back response for lag>=2 (1-back is
    left uncorrected), mirroring analyze_subject."""
    r = g[["resp_rel_1bk", "err"]].dropna()
    resp_fit = (fit_dog(-r["resp_rel_1bk"].values.astype(float),
                        -r["err"].values.astype(float))
                if len(r) >= MIN_TRIALS else None)
    out = {}
    for n in NBACKS:
        col = f"rel_{n}bk"
        sub = g[[col, "err", "resp_rel_1bk"]].dropna()
        if len(sub) < MIN_TRIALS:
            out[n] = (None, None)
            continue
        x = -sub[col].values.astype(float)
        y = -sub["err"].values.astype(float)
        _, cu = bin_average(x, y, 18)
        if n == 1 or resp_fit is None:
            cf = cu
        else:
            xr = -sub["resp_rel_1bk"].values.astype(float)
            _, cf = bin_average(x, y - dog_func(xr, *resp_fit), 18)
        out[n] = (cu, cf)
    return out


def analyze(csv_path, subj_excl=None):
    df = pd.read_csv(csv_path)

    # subject exclusion by mean absolute error (only if requested)
    n_total = df["uid"].nunique()
    if subj_excl is not None:
        keep = [uid for uid, g in df.groupby("uid")
                if g["err"].abs().mean() <= subj_excl]
        df = df[df["uid"].isin(keep)].copy()
    n_subj = df["uid"].nunique()

    # trial exclusion
    df = df[df["err"].abs() <= TRIAL_EXCL].copy()

    amps_unc = {n: [] for n in NBACKS}
    amps_cor = {n: [] for n in NBACKS}
    curves = {n: [] for n in NBACKS}
    for uid, g in df.groupby("uid"):
        res = analyze_subject(g)
        if res is None:
            continue
        for n in NBACKS:
            au, ac = res[n]
            if np.isfinite(au):
                amps_unc[n].append(au)
            if np.isfinite(ac):
                amps_cor[n].append(ac)
            c = binned_curve(g, n)
            if c is not None:
                curves[n].append(c)
    return amps_unc, amps_cor, curves, n_subj, n_total


def dataset_profiles(csv_path, subj_excl=None):
    """Reusable summary of a human-format CSV for downstream scripts
    (temporal-horizon rebuild, figures).

    Returns a dict with:
      'per_subject_unc'   : {n: np.array} aligned per-subject uncorrected amps
      'per_subject_final' : {n: np.array} aligned per-subject final amps
                            (1-back uncorrected; lag>=2 autocorr-corrected)
      'curves'            : {n: mean binned DoG curve across subjects}
      'centers'           : bin centers for the curves
      'n_subj'            : number of subjects contributing aligned amps

    "Aligned" = only subjects with all three n-back amplitudes finite (for both
    the uncorrected and final profiles), so per-n arrays share a subject axis and
    can be jointly bootstrap-resampled.
    """
    df = pd.read_csv(csv_path)
    return profiles_from_df(df, subj_excl)


def profiles_from_df(df, subj_excl=None):
    """Same as dataset_profiles but takes an in-memory human-format DataFrame,
    so model-generated trial tables run through the identical code path as human
    data. Also returns mean uncorrected AND final (corrected) DoG curves."""
    if subj_excl is not None:
        keep = [uid for uid, g in df.groupby("uid")
                if g["err"].abs().mean() <= subj_excl]
        df = df[df["uid"].isin(keep)].copy()
    df = df[df["err"].abs() <= TRIAL_EXCL].copy()

    unc = {n: [] for n in NBACKS}
    fin = {n: [] for n in NBACKS}
    fin_b = {n: [] for n in NBACKS}          # per-subject final DoG width b
    curves_u = {n: [] for n in NBACKS}
    curves_f = {n: [] for n in NBACKS}
    for uid, g in df.groupby("uid"):
        fits = subject_fits(g)
        if fits is None:
            continue
        final_fit = {1: fits[1][0], 2: fits[2][1], 3: fits[3][1]}
        u = {n: (fits[n][0][0] if fits[n][0] else np.nan) for n in NBACKS}
        f = {n: (final_fit[n][0] if final_fit[n] else np.nan) for n in NBACKS}
        if not all(np.isfinite(list(u.values()) + list(f.values()))):
            continue
        cur = subject_curves(g)
        for n in NBACKS:
            unc[n].append(u[n])
            fin[n].append(f[n])
            fin_b[n].append(final_fit[n][1])
            cu, cf = cur[n]
            if cu is not None:
                curves_u[n].append(cu)
            if cf is not None:
                curves_f[n].append(cf)

    def _mean_curve(d):
        return {n: (np.nanmean(np.array(d[n]), axis=0) if d[n] else None)
                for n in NBACKS}

    # Display DoG per n-back: amplitude = group-mean per-subject amplitude (so the
    # drawn curve matches the reported/table value), width = median per-subject b
    # (robust). This avoids fitting a spurious wide repulsion to a noisy aggregate
    # curve, which happens for small, noisy lag>=2 effects (e.g. Pascucci 2/3-back).
    final_fit_disp = {}
    for n in NBACKS:
        a = np.nanmean(fin[n]) if fin[n] else np.nan
        b = np.nanmedian(fin_b[n]) if fin_b[n] else np.nan
        final_fit_disp[n] = (a, b) if np.isfinite(a) and np.isfinite(b) else None

    return {
        "per_subject_unc": {n: np.array(unc[n]) for n in NBACKS},
        "per_subject_final": {n: np.array(fin[n]) for n in NBACKS},
        "curves": _mean_curve(curves_u),          # uncorrected (back-compat)
        "curves_unc": _mean_curve(curves_u),
        "curves_final": _mean_curve(curves_f),
        "final_fit": final_fit_disp,              # (a, b) display DoG per n-back
        "centers": np.linspace(-90, 90, 18),
        "n_subj": len(unc[NBACKS[0]]),
    }


def final_profile(amps_unc, amps_cor):
    """The autocorrelation-aware n-back profile: 1-back is reported uncorrected
    (it cannot be response-residualised), lags >= 2 are corrected."""
    return {1: amps_unc[1], 2: amps_cor[2], 3: amps_cor[3]}


def fit_tau(amps_by_n):
    """Fit A_n = A_1 * exp(-(n-1)/tau) to mean amplitudes; return tau or None.
    Requires monotone-positive means to be well-posed."""
    means = np.array([np.mean(amps_by_n[n]) if amps_by_n[n] else np.nan
                      for n in NBACKS])
    if not np.all(np.isfinite(means)) or np.any(means <= 0):
        return None
    ns = np.array(NBACKS, dtype=float)
    # log-linear fit: log A_n = log A_1 - (n-1)/tau
    slope, intercept = np.polyfit(ns - 1, np.log(means), 1)
    if slope >= 0:
        return None
    return float(-1.0 / slope)


def _mean_sem(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return 0.0, 0.0
    return v.mean(), (stats.sem(v) if v.size > 1 else 0.0)


def make_figure(amps_unc, amps_final, out_path, title):
    plt = config.matplotlib_setup()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), clear=True)
    x = np.arange(len(NBACKS))
    w = 0.38
    mu, su = zip(*[_mean_sem(amps_unc[n]) for n in NBACKS])
    mf, sf = zip(*[_mean_sem(amps_final[n]) for n in NBACKS])
    ax.bar(x - w / 2, mu, w, yerr=su, capsize=4,
           label="uncorrected", color="#c44e52")
    ax.bar(x + w / 2, mf, w, yerr=sf, capsize=4,
           label=r"autocorr-aware (lag$\geq$2)", color="#4c72b0")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NBACKS])
    ax.set_xlabel("n-back")
    ax.set_ylabel(r"SD amplitude ($^\circ$)")
    ax.set_title(str(title).replace("_", "-"))   # underscore breaks usetex
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=config.HUMAN_CSV)
    p.add_argument("--tag", default="human",
                   help="label used in output filenames")
    p.add_argument("--subj_excl", type=float, default=None,
                   help="exclude subjects with mean|err| above this (deg). "
                        "Use 39.2 to reproduce the in-house bhv_hum pipeline; "
                        "omit for public data (no ad-hoc exclusion).")
    args = p.parse_args()

    os.makedirs(config.FIGS_DIR, exist_ok=True)
    amps_unc, amps_cor, curves, n_subj, n_total = analyze(args.csv, args.subj_excl)
    amps_final = final_profile(amps_unc, amps_cor)

    tau_unc = fit_tau(amps_unc)
    tau_final = fit_tau(amps_final)

    # ---- table ----
    rows = []
    print("\n" + "=" * 92)
    print(f"n-back SD amplitude  ({args.tag})   N={n_subj}"
          + (f" / {n_total}" if args.subj_excl is not None else ""))
    print("  'final' = 1-back uncorrected + lag>=2 autocorrelation-corrected "
          "(Ma et al. 2025)")
    print("-" * 92)
    print(f"{'n-back':>6} | {'uncorrected mean±SEM':>22} {'p':>7} | "
          f"{'corrected mean±SEM':>22} {'p':>7} | {'final':>8}")
    print("-" * 92)
    for n in NBACKS:
        au = np.array(amps_unc[n])
        ac = np.array([x for x in amps_cor[n] if np.isfinite(x)])
        mu, su = _mean_sem(au)
        _, pu = _one_sided_p(au)
        af = np.array([x for x in amps_final[n] if np.isfinite(x)])
        mf, sf = _mean_sem(af)
        _, pf = _one_sided_p(af)
        if n == 1:
            mc_s, pc = "        n/a", np.nan
            mc = sc = np.nan
        else:
            mc, sc = _mean_sem(ac)
            _, pc = _one_sided_p(ac)
            mc_s = f"{mc:>10.3f}±{sc:<10.3f}"
        pc_s = "    n/a" if not np.isfinite(pc) else f"{pc:>7.3f}"
        print(f"{n:>6} | {mu:>10.3f}±{su:<10.3f} {pu:>7.3f} | "
              f"{mc_s} {pc_s} | {mf:>8.3f}")
        rows.append({"nback": n,
                     "unc_mean": float(mu), "unc_sem": float(su),
                     "unc_p_onesided": float(pu), "unc_n": int(au.size),
                     "cor_mean": float(mc), "cor_sem": float(sc),
                     "cor_p_onesided": float(pc), "cor_n": int(ac.size),
                     "final_mean": float(mf), "final_sem": float(sf),
                     "final_p_onesided": float(pf)})
    print("-" * 92)
    print(f"tau (exp decay):  uncorrected={fmt(tau_unc,2)}   "
          f"final(autocorr-aware)={fmt(tau_final,2)}")
    print(f"monotonic decreasing?  uncorrected="
          f"{_is_monotone([r['unc_mean'] for r in rows])}   "
          f"final={_is_monotone([r['final_mean'] for r in rows])}")
    print("=" * 92)

    # ---- figure ----
    fig_path = os.path.join(config.FIGS_DIR, f"nback_autocorr_{args.tag}.pdf")
    make_figure(amps_unc, amps_final, fig_path, args.tag)
    print(f"[fig] wrote {fig_path}")

    # ---- persist ----
    data = {
        "meta": {
            "csv": args.csv, "tag": args.tag,
            "n_subjects": int(n_subj), "n_subjects_total": int(n_total),
            "subj_excl_mean_abs_err_deg": args.subj_excl,
            "trial_excl_abs_deg": TRIAL_EXCL, "min_trials": MIN_TRIALS,
            "correction": "Ma et al. 2025: residualise 1-back response DoG, "
                          "then fit n-back stimulus DoG on residuals",
            "test": "one-sample t-test vs 0, one-sided",
            "tau_uncorrected": tau_unc, "tau_final": tau_final,
            "final_profile": "1-back uncorrected; lag>=2 autocorr-corrected",
            "monotone_uncorrected": _is_monotone([r["unc_mean"] for r in rows]),
            "monotone_final": _is_monotone([r["final_mean"] for r in rows]),
            "figure": fig_path,
        },
        "table": rows,
        "amplitudes": {
            "uncorrected": {str(n): [float(x) for x in amps_unc[n]] for n in NBACKS},
            "corrected": {str(n): [float(x) for x in amps_cor[n]
                                   if np.isfinite(x)] for n in NBACKS},
        },
    }
    save_json(f"human_sd_{args.tag}.json", data, config.RESULTS_DIR)

    md = [f"# n-back serial dependence — {args.tag}\n",
          f"- Source: `{args.csv}`",
          f"- N = {n_subj} subjects"
          + (f" (of {n_total}; excluded mean|err|>{args.subj_excl}°)"
             if args.subj_excl is not None else " (no subject exclusion)"),
          f"- Trials with |err|>{TRIAL_EXCL}° dropped; ≥{MIN_TRIALS} trials/cell.",
          "- Correction = Ma et al. (2025): residualise 1-back *response* DoG, "
          "then refit lag≥2 *stimulus* DoG (1-back cannot be corrected — it is "
          "collinear with the 1-back response).",
          f"- τ (A_n = A_1·exp(−(n−1)/τ)): uncorrected {fmt(tau_unc,2)}, "
          f"final {fmt(tau_final,2)}.",
          f"- Monotonic decreasing? uncorrected="
          f"{_is_monotone([r['unc_mean'] for r in rows])}, "
          f"final={_is_monotone([r['final_mean'] for r in rows])}.\n",
          "| n-back | Uncorr mean±SEM (°) | Uncorr p | Corr mean±SEM (°) | Corr p "
          "| **Final** mean (°) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        cor = "n/a" if r["nback"] == 1 else \
            f"{fmt(r['cor_mean'],3)}±{fmt(r['cor_sem'],3)}"
        corp = "n/a" if r["nback"] == 1 else fmt(r["cor_p_onesided"], 3)
        md.append(f"| {r['nback']} | {fmt(r['unc_mean'],3)}±{fmt(r['unc_sem'],3)} "
                  f"| {fmt(r['unc_p_onesided'],3)} | {cor} | {corp} "
                  f"| **{fmt(r['final_mean'],3)}** |")
    save_markdown(f"human_sd_{args.tag}.md", "\n".join(md) + "\n", config.RESULTS_DIR)
    print(f"[results] wrote {config.RESULTS_DIR}/human_sd_{args.tag}.json + .md")
