"""Gabor patch generation and orientation estimation by template matching."""

import numpy as np


def generate_gabor(size=64, sigma=10.0, theta=0.0, lambda_=10.0, psi=0.0, gamma=1.0):
    """Standard Gabor patch, normalized to [0, 1]. theta in radians (orientation)."""
    half = size // 2
    y, x = np.meshgrid(
        np.arange(size) - half,
        np.arange(size) - half,
        indexing="ij",
    )
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2)) * np.cos(
        2 * np.pi * x_theta / lambda_ + psi
    )
    gb = gb - gb.min()
    if gb.max() > 0:
        gb = gb / gb.max()
    return gb.astype(np.float32)


_TEMPLATE_CACHE = {}


def _get_templates(size, sigma, lambda_, psi, gamma):
    key = (size, sigma, lambda_, psi, gamma)
    if key not in _TEMPLATE_CACHE:
        templates = np.zeros((180, size * size), np.float32)
        for deg in range(180):
            theta = np.deg2rad(deg)
            g = generate_gabor(size, sigma, theta, lambda_, psi, gamma).ravel()
            g = g - g.mean()
            std = g.std()
            if std > 0:
                g = g / std
            templates[deg] = g
        _TEMPLATE_CACHE[key] = templates
    return _TEMPLATE_CACHE[key]


def estimate_orientation(image, size=64, sigma=10.0, lambda_=10.0, psi=0.0, gamma=1.0):
    """Return the orientation (radians) of the best-correlating Gabor template."""
    templates = _get_templates(size, sigma, lambda_, psi, gamma)
    img = np.asarray(image, dtype=np.float32).ravel()
    img = img - img.mean()
    std = img.std()
    if std > 0:
        img = img / std
    corr = templates @ img / img.size  # normalized correlation per template
    best_deg = int(np.argmax(corr))
    return np.deg2rad(best_deg)
