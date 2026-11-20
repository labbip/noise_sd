"""Shared paths and constants. Paths can be overridden via environment variables."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("SDV_DATA_DIR", os.path.join(ROOT, "data"))
VIDEO_DIR = os.environ.get("SDV_VIDEO_DIR", os.path.join(DATA_DIR, "CharadesEgo_v1_480"))
ANNOT_CSV = os.environ.get(
    "SDV_ANNOT_CSV", os.path.join(DATA_DIR, "CharadesEgo", "CharadesEgo_v1_train.csv")
)
HUMAN_CSV = os.environ.get("SDV_HUMAN_CSV", os.path.join(DATA_DIR, "bhv_hum.csv"))

MODELS_DIR = os.path.join(ROOT, "models")
SUBJECTS_DIR = os.path.join(MODELS_DIR, "subjects")
FIGS_DIR = os.path.join(ROOT, "figs")
RESULTS_DIR = os.path.join(ROOT, "results")

# training defaults (paper Methods 2.2)
SEQ_LEN = 16
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
MAX_VIDEOS = 500
IMG_SIZE = 64

NOISE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Gabor evaluation parameters
GABOR = dict(size=64, sigma=10.0, lambda_=10.0, psi=0.0, gamma=1.0)


def check_data():
    problems = []
    if not os.path.exists(VIDEO_DIR):
        problems.append(f"VIDEO_DIR missing: {VIDEO_DIR}")
    if not os.path.exists(ANNOT_CSV):
        problems.append(f"ANNOT_CSV missing: {ANNOT_CSV}")
    if not os.path.exists(HUMAN_CSV):
        problems.append(f"HUMAN_CSV missing: {HUMAN_CSV}")
    return problems


def matplotlib_setup(usetex=True):
    import matplotlib

    # Figures are only saved to PDF, so always use the non-interactive backend
    # (no GUI toolkit such as Qt is required).
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "Helvetica",
            "text.latex.preamble": "\\usepackage{sfmath}",
            "font.size": 20,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    return plt
