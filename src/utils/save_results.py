"""Helpers to persist numerical results / test statistics for downstream use.

Each evaluation script writes:
  - a machine-readable JSON (full precision, nested structure)
  - a human-readable Markdown summary (tables, for a paper-writing agent)
Both go under config.RESULTS_DIR.
"""

import json
import os
from datetime import datetime

import numpy as np


def _to_native(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_json(name, data, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data": _to_native(data),
    }
    path = os.path.join(results_dir, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return path


def save_markdown(name, text, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    header = f"<!-- generated {datetime.now().isoformat(timespec='seconds')} -->\n\n"
    with open(path, "w") as f:
        f.write(header + text)
    return path


def fmt(v, nd=4):
    """Format a possibly-None/NaN number for Markdown tables."""
    if v is None:
        return "n/a"
    try:
        if np.isnan(v) or np.isinf(v):
            return "n/a"
    except (TypeError, ValueError):
        return str(v)
    return f"{v:.{nd}f}"
