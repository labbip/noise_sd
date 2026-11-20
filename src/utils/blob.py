"""2D Gaussian dot generation and sub-pixel centroid estimation."""

import numpy as np


def generate_blob(size=64, pos=(32.0, 32.0), sigma=8.0):
    """2D Gaussian blob at pos=(x, y), peak normalized to 1.0."""
    x0, y0 = pos
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    return g.astype(np.float32)


def estimate_centroid(image, threshold=0.1):
    """Weighted (sub-pixel) centroid of pixels above threshold*max. Returns (x, y)."""
    img = np.asarray(image, dtype=np.float32)
    thr = img.max() * threshold
    mask = img >= thr
    if not mask.any():
        h, w = img.shape
        return (w / 2.0, h / 2.0)
    y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
    w = img * mask
    total = w.sum()
    cx = (w * x).sum() / total
    cy = (w * y).sum() / total
    return (float(cx), float(cy))
