"""Derivative-of-Gaussian (DoG) fitting and serial-dependence utilities."""

import numpy as np
from scipy.optimize import curve_fit


def dog_func(x, a, b):
    """First derivative of a Gaussian.

    a is the SD amplitude (strength); peak position is Delta = 1/(b*sqrt(2)).
    """
    return a * np.sqrt(2 * np.e) * b * x * np.exp(-((b * x) ** 2))


# Bounds keep fits well-posed: amplitude |a| <= A_MAX (deg) and the DoG peak
# position Delta = 1/(b*sqrt(2)) constrained to [DELTA_MIN, DELTA_MAX] degrees.
# Without a two-sided bound on b, noisy per-subject data yields degenerate
# narrow-spike fits with huge amplitudes.
A_MAX = 45.0
DELTA_MIN = 1.0
DELTA_MAX = 90.0


def fit_dog(x, y, a_max=A_MAX, delta_min=DELTA_MIN, delta_max=DELTA_MAX):
    """Fit dog_func; returns (a, b) or None on failure."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    b_min = 1.0 / (delta_max * np.sqrt(2))
    b_max = 1.0 / (delta_min * np.sqrt(2))
    try:
        popt, _ = curve_fit(
            dog_func,
            x,
            y,
            p0=[1.0, 1.0 / (20.0 * np.sqrt(2))],
            bounds=([-a_max, b_min], [a_max, b_max]),
            maxfev=100000,
        )
        return popt[0], popt[1]
    except (RuntimeError, ValueError):
        return None


def get_angle_diff(a, b):
    """Signed difference a - b wrapped to (-90, 90]."""
    d = (a - b) % 180.0
    if np.isscalar(d):
        if d > 90.0:
            d -= 180.0
        return d
    d = np.where(d > 90.0, d - 180.0, d)
    return d


def bin_average(x, y, n_bins=18, lo=-90.0, hi=90.0):
    """Average y within n_bins equal bins over [lo, hi]. Returns (centers, means)."""
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x = np.asarray(x)
    y = np.asarray(y)
    means = np.full(n_bins, np.nan)
    idx = np.digitize(x, edges) - 1
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            means[b] = np.nanmean(y[sel])
    return centers, means


def permutation_test_sd(rel, err, n=100, n_bins=18):
    """p-value: fraction of shuffled DoG amplitudes with |a| >= observed |a|."""
    rel = np.asarray(rel, dtype=float)
    err = np.asarray(err, dtype=float)
    centers, means = bin_average(rel, err, n_bins)
    fit = fit_dog(centers, means)
    if fit is None:
        return 1.0, None
    obs = abs(fit[0])
    count = 0
    valid = 0
    for _ in range(n):
        perm = np.random.permutation(rel)
        c, m = bin_average(perm, err, n_bins)
        f = fit_dog(c, m)
        if f is None:
            continue
        valid += 1
        if abs(f[0]) >= obs:
            count += 1
    if valid == 0:
        return 1.0, fit
    return count / valid, fit
