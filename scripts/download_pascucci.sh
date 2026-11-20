#!/usr/bin/env bash
# Download the Pascucci et al. (2019) public orientation-adjustment data and
# convert it to the analysis format (data/pascucci2019/bhv_pascucci_pool12.csv).
#
# Dataset: Pascucci et al. (2019), "Laws of concatenated perception: Vision goes
# for novelty, decisions for perseverance", PLoS Biology 17(3):e3000144.
# Raw data (CC): https://doi.org/10.5281/zenodo.2544946
#
# We use the two baseline orientation-adjustment experiments (Exp1 + Exp2),
# pooled into N=20 observers, for the independent human comparison (Fig 7b/8,
# Table 1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
DATA_DIR="${SDV_DATA_DIR:-$ROOT/data}"
OUT="$DATA_DIR/pascucci2019"
mkdir -p "$OUT"
cd "$OUT"

ZENODO="https://zenodo.org/api/records/2544946/files"

dl() {  # filename
  if [ -f "$1" ]; then echo "[skip] $1 exists"; return; fi
  echo "[download] $1"
  if command -v wget >/dev/null 2>&1; then wget -q -O "$1" "$ZENODO/$1/content";
  else curl -sL -o "$1" "$ZENODO/$1/content"; fi
}

for n in 1 2; do dl "Experiment${n}_rawdata.csv"; done

echo "[convert] Exp1 + Exp2 -> bhv_pascucci_pool12.csv"
cd "$ROOT"
uv run python convert_pascucci.py --exp 1,2

echo "[done] $OUT/bhv_pascucci_pool12.csv"
