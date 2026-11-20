"""Convert Pascucci et al. (2019) raw orientation-adjustment data to the
human-CSV format used by the serial-dependence pipeline.

Pascucci et al. 2019, "Laws of concatenated perception", PLoS Biology,
raw data: https://doi.org/10.5281/zenodo.2544946

Only the orientation-adjustment experiments are supported:
  Exp1 : baseline (10 obs x 560 trials, 18 discrete orientations)  [canonical]
  Exp2 : continuous orientations (10 obs x 440)
  Exp3 : has a `condition` column (15 obs x <=600)  -> block reset per condition
  Exp6 : contrast manipulation (15 obs x <=400)

Output columns match data/bhv_hum.csv semantics so the same analysis runs on
both.  Sign conventions (the analysis uses x = -rel and y = -err, so a positive
DoG amplitude denotes attraction toward the previous stimulus):
  err          = orientation - reported            (-err = response - truth,   + = attraction)
  rel_Nbk      = ang(current_ori  - prev_N_ori)     (-rel = previous - current)
  resp_rel_Nbk = ang(current_ori  - prev_N_reported) (nuisance regressor for the
                 response-autocorrelation correction; -resp_rel = prev_resp - current)

n-back columns reset at observer boundaries (and condition boundaries for Exp3),
so they are NaN for the first N trials of every sequence -- exactly as the
in-house bhv_hum.csv encodes block starts.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.utils.dog import get_angle_diff

NBACKS = [1, 2, 3]


def _seq_columns(ori, resp):
    """Given ordered stimulus/response arrays for one sequence, return the
    err and per-nback rel / resp_rel columns (NaN where n-back is undefined)."""
    ori = np.asarray(ori, dtype=float)
    resp = np.asarray(resp, dtype=float)
    n = len(ori)
    err = get_angle_diff(ori, resp)  # truth - response
    out = {"err": err}
    for k in NBACKS:
        rel = np.full(n, np.nan)
        resp_rel = np.full(n, np.nan)
        if n > k:
            rel[k:] = get_angle_diff(ori[k:], ori[:-k])
            resp_rel[k:] = get_angle_diff(ori[k:], resp[:-k])
        out[f"rel_{k}bk"] = rel
        out[f"resp_rel_{k}bk"] = resp_rel
    return out


def convert(exp_path, exp_tag, obs_col="observer", ori_col="orientation",
            resp_col="reported", block_col=None):
    """Convert one experiment. uid is namespaced by experiment so observers
    from different experiments never collide when pooled."""
    df = pd.read_csv(exp_path)
    rows = []
    group_cols = [obs_col] + ([block_col] if block_col else [])
    for keys, g in df.groupby(group_cols, sort=False):
        uid = keys[0] if isinstance(keys, tuple) else keys
        cols = _seq_columns(g[ori_col].values, g[resp_col].values)
        rec = pd.DataFrame({
            "uid": [f"pascucci_e{exp_tag}_obs{int(uid)}"] * len(g),
            "ori": g[ori_col].values,
            "resp": g[resp_col].values,
            **cols,
        })
        rows.append(rec)
    return pd.concat(rows, ignore_index=True)


EXPERIMENTS = {
    "1": dict(fname="Experiment1_rawdata.csv"),
    "2": dict(fname="Experiment2_rawdata.csv"),
    "3": dict(fname="Experiment3_rawdata.csv", block_col="condition"),
    "6": dict(fname="Experiment6_rawdata.csv"),
}


def _sanity(df, label):
    sub = df.dropna(subset=["rel_1bk", "err"])
    x = -sub["rel_1bk"].values
    y = -sub["err"].values
    near = np.abs(x) <= 30
    print(f"[convert] {label}: {len(df)} trials, {df['uid'].nunique()} observers. "
          f"near-zero mean bias in prev direction = "
          f"{np.mean(np.sign(x[near]) * y[near]):.3f} deg (+=attraction)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default="data/pascucci2019")
    p.add_argument("--exp", default="1",
                   help="experiment(s) to convert, comma-separated "
                        "(e.g. '1' or '1,2'). Multiple -> pooled CSV.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    exps = [e.strip() for e in args.exp.split(",") if e.strip()]
    for e in exps:
        if e not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment '{e}'; "
                             f"choose from {sorted(EXPERIMENTS)}")

    parts = []
    for e in exps:
        spec = EXPERIMENTS[e]
        src = os.path.join(args.src_dir, spec["fname"])
        d = convert(src, exp_tag=e, block_col=spec.get("block_col"))
        _sanity(d, f"Exp{e}")
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    if len(exps) == 1:
        default_out = f"bhv_pascucci_exp{exps[0]}.csv"
    else:
        default_out = f"bhv_pascucci_pool{''.join(exps)}.csv"
    out = args.out or os.path.join(args.src_dir, default_out)
    df.to_csv(out, index=False)
    if len(exps) > 1:
        _sanity(df, "POOLED " + "+".join(exps))
    print(f"[convert] wrote {out}")
